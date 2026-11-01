// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The fitsio-pure backend of [`super::uvfits::UvfitsWriter`].
//!
//! The random-groups primary HDU is streamed to the file with fitsio-pure's
//! [`GroupsWriter`], a group at a time, so memory use doesn't grow with the
//! file, as with cfitsio. The antenna table follows it in the same stream.
//! Header keys are kept as cards until their HDU's header is written; the
//! key writers update a key that is already there, as cfitsio's `ffuky*` do.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use fitsio_pure::{
    bintable::{
        binary_type_byte_size, parse_tform_binary, serialize_binary_table, BinaryColumnData,
        BinaryColumnDescriptor, BinaryColumnType,
    },
    compat::errors::Error,
    groups::{groups_header, GroupsWriter},
    header::{serialize_header, Card},
    value::Value,
};

type Result<T> = std::result::Result<T, Error>;

/// A uvfits file being written.
pub(crate) struct UvfitsFile {
    state: State,
}

enum State {
    /// The primary header, until its groups start.
    PrimaryHeader {
        out: BufWriter<File>,
        cards: Vec<Card>,
    },
    /// Streaming the random groups.
    Groups(GroupsWriter<BufWriter<File>>),
    /// The antenna table, written out on close.
    Table {
        out: BufWriter<File>,
        cards: Vec<Card>,
        columns: Vec<BinaryColumnDescriptor>,
        data: Option<(Vec<BinaryColumnData>, usize)>,
    },
    Closed,
}

impl UvfitsFile {
    /// Create (or truncate) the file at `path`.
    pub(crate) fn create(path: &Path) -> Result<UvfitsFile> {
        Ok(UvfitsFile {
            state: State::PrimaryHeader {
                out: BufWriter::new(File::create(path)?),
                cards: vec![],
            },
        })
    }

    /// The cards of the header being written.
    fn cards(&mut self) -> Result<&mut Vec<Card>> {
        match &mut self.state {
            State::PrimaryHeader { cards, .. } | State::Table { cards, .. } => Ok(cards),
            _ => Err(Error::Message(
                "uvfits header keys can't be written now".to_string(),
            )),
        }
    }

    /// Write the primary header for `gcount` random groups of `pcount`
    /// parameters and a float data array of axes `naxes` (`NAXIS2` onwards):
    /// the cards cfitsio's `ffphpr` writes, then the keys written so far in
    /// the order they were written, as cfitsio leaves them.
    pub(crate) fn start_groups(
        &mut self,
        naxes: &[usize],
        pcount: usize,
        gcount: usize,
    ) -> Result<()> {
        let State::PrimaryHeader { out, cards } = std::mem::replace(&mut self.state, State::Closed)
        else {
            return Err(Error::Message("uvfits groups already started".to_string()));
        };
        // ffphpr's cards, with no parameters yet; the parameters' PTYPEn,
        // PSCALn and PZEROn are among `cards`.
        let mut header = groups_header(-32, naxes, &[], gcount)?;
        if let Some(card) = header.iter_mut().find(|c| c.keyword_str() == "PCOUNT") {
            card.value = Some(Value::Integer(pcount as i64));
        }
        header.extend(cards);
        self.state = State::Groups(GroupsWriter::with_header(out, &header)?);
        Ok(())
    }

    /// Write the next group: its parameters, then its data array, as cfitsio's
    /// `ffpgpe` writes them from the group's first element.
    pub(crate) fn write_group(&mut self, values: &[f32]) -> Result<()> {
        match &mut self.state {
            State::Groups(groups) => Ok(groups.write_group(values)?),
            _ => Err(Error::Message(
                "uvfits groups aren't being written".to_string(),
            )),
        }
    }

    /// Finish the groups and start a binary table extension with columns
    /// `names`, `formats` (`TFORMn`) and `units` (`TUNITn`, none when empty),
    /// with the header cfitsio's `ffcrtb` writes.
    pub(crate) fn start_table(
        &mut self,
        names: &[&str],
        formats: &[&str],
        units: &[&str],
        extname: &str,
    ) -> Result<()> {
        let State::Groups(groups) = std::mem::replace(&mut self.state, State::Closed) else {
            return Err(Error::Message(
                "uvfits groups aren't being written".to_string(),
            ));
        };
        let out = groups.finish()?;

        let mut columns = Vec::with_capacity(names.len());
        for (name, format) in names.iter().zip(formats) {
            let (repeat, col_type) = parse_tform_binary(format)?;
            columns.push(BinaryColumnDescriptor {
                name: Some(name.to_string()),
                repeat,
                byte_width: repeat * binary_type_byte_size(&col_type),
                col_type,
                tdim: None,
            });
        }
        let width: usize = columns.iter().map(|c| c.byte_width).sum();
        let mut cards = vec![
            card(
                "XTENSION",
                Value::String("BINTABLE".to_string()),
                "binary table extension",
            )?,
            card("BITPIX", Value::Integer(8), "8-bit bytes")?,
            card("NAXIS", Value::Integer(2), "2-dimensional binary table")?,
            card(
                "NAXIS1",
                Value::Integer(width as i64),
                "width of table in bytes",
            )?,
            card("NAXIS2", Value::Integer(0), "number of rows in table")?,
            card("PCOUNT", Value::Integer(0), "size of special data area")?,
            card(
                "GCOUNT",
                Value::Integer(1),
                "one data group (required keyword)",
            )?,
            card(
                "TFIELDS",
                Value::Integer(columns.len() as i64),
                "number of fields in each row",
            )?,
        ];
        for (i, ((name, format), unit)) in names.iter().zip(formats).zip(units).enumerate() {
            let n = i + 1;
            cards.push(card(
                &format!("TTYPE{n}"),
                Value::String(name.to_string()),
                &format!("label for field {n:3}"),
            )?);
            cards.push(card(
                &format!("TFORM{n}"),
                Value::String(format.to_string()),
                &format!(
                    "data format of field: {}",
                    tform_description(&columns[i].col_type)
                ),
            )?);
            if !unit.is_empty() {
                cards.push(card(
                    &format!("TUNIT{n}"),
                    Value::String(unit.to_string()),
                    "physical unit of field",
                )?);
            }
        }
        cards.push(card(
            "EXTNAME",
            Value::String(extname.to_string()),
            "name of this binary table extension",
        )?);

        self.state = State::Table {
            out,
            cards,
            columns,
            data: None,
        };
        Ok(())
    }

    /// Set the table's `num_rows` rows, a column at a time in column order.
    pub(crate) fn write_table(
        &mut self,
        columns: Vec<BinaryColumnData>,
        num_rows: usize,
    ) -> Result<()> {
        match &mut self.state {
            State::Table { data, .. } => {
                *data = Some((columns, num_rows));
                Ok(())
            }
            _ => Err(Error::Message(
                "no uvfits table is being written".to_string(),
            )),
        }
    }

    /// Write out whatever is pending and flush the file, as cfitsio's
    /// `ffclos` does: unwritten groups are zero-filled and the data padded.
    pub(crate) fn close(&mut self) -> Result<()> {
        match std::mem::replace(&mut self.state, State::Closed) {
            State::PrimaryHeader { .. } | State::Closed => Ok(()),
            State::Groups(groups) => Ok(groups.finish()?.flush()?),
            State::Table {
                mut out,
                mut cards,
                columns,
                data,
            } => {
                let (data, num_rows) = data.unwrap_or_default();
                if let Some(naxis2) = cards.iter_mut().find(|c| c.keyword_str() == "NAXIS2") {
                    naxis2.value = Some(Value::Integer(num_rows as i64));
                }
                out.write_all(&serialize_header(&cards)?)?;
                if num_rows > 0 {
                    out.write_all(&serialize_binary_table(&columns, &data, num_rows)?)?;
                }
                Ok(out.flush()?)
            }
        }
    }
}

fn card(keyword: &str, value: Value, comment: &str) -> Result<Card> {
    let mut card = Card::new(keyword, value)?;
    card.comment = Some(comment.to_string());
    Ok(card)
}

/// The type description cfitsio's `ffcrtb` gives each `TFORMn`.
fn tform_description(col_type: &BinaryColumnType) -> &'static str {
    match col_type {
        BinaryColumnType::Logical => "1-byte LOGICAL",
        BinaryColumnType::Bit => "BIT",
        BinaryColumnType::Byte => "BYTE",
        BinaryColumnType::Short => "2-byte INTEGER",
        BinaryColumnType::Int => "4-byte INTEGER",
        BinaryColumnType::Long => "8-byte INTEGER",
        BinaryColumnType::Float => "4-byte REAL",
        BinaryColumnType::Double => "8-byte DOUBLE",
        BinaryColumnType::ComplexFloat => "COMPLEX",
        BinaryColumnType::ComplexDouble => "DOUBLE COMPLEX",
        BinaryColumnType::Ascii => "ASCII Character",
        BinaryColumnType::VarArrayP(_) | BinaryColumnType::VarArrayQ(_) => "variable length array",
    }
}

/// Set `keyname`, as cfitsio's `ffuky*` do: a key that is already there is
/// updated in place, keeping its comment unless a new one is given.
pub(crate) fn update_key(
    fptr: &mut UvfitsFile,
    keyname: &str,
    value: Value,
    comment: Option<&str>,
) -> Result<()> {
    let cards = fptr.cards()?;
    match cards.iter_mut().find(|c| c.keyword_str() == keyname) {
        Some(existing) => {
            existing.value = Some(value);
            if let Some(comment) = comment {
                existing.comment = Some(comment.to_string());
            }
        }
        None => {
            let mut card = Card::new(keyname, value)?;
            card.comment = comment.map(str::to_string);
            cards.push(card);
        }
    }
    Ok(())
}

/// Append `COMMENT` or `HISTORY` cards holding `text`, as cfitsio's `ffpcom`
/// and `ffphis` do: one card per 72 bytes, with anything but printable ASCII
/// written as a blank.
pub(crate) fn write_commentary(fptr: &mut UvfitsFile, keyword: &[u8; 8], text: &str) -> Result<()> {
    let cards = fptr.cards()?;
    for chunk in text.as_bytes().chunks(72) {
        let text: String = chunk
            .iter()
            .map(|&b| {
                if (b' '..=b'~').contains(&b) {
                    b as char
                } else {
                    ' '
                }
            })
            .collect();
        cards.push(Card {
            keyword: *keyword,
            value: None,
            comment: Some(text),
        });
    }
    Ok(())
}
