// SPDX-License-Identifier: AGPL-3.0-only

//! Wire-format tests for the EP vision payload.
//!
//! The header is the only part of the exchange a host can exercise: the bulk
//! row broadcast needs NCCL and the device methods need a live
//! `TransformerModel`. Everything the receiver would otherwise take on trust
//! — the word count, the grids, and `n_rows` agreeing with them — is decided
//! here.

use super::{decode_vision_header, encode_vision_header, stage_row_bases};

fn rows_of(grids: &[(usize, usize)]) -> usize {
    grids.iter().map(|(h, w)| h * w).sum()
}

#[test]
fn the_header_round_trips_for_one_two_and_five_images() {
    for grids in [
        vec![(2usize, 3usize)],
        vec![(2, 3), (4, 4)],
        vec![(1, 1), (2, 2), (3, 5), (7, 2), (4, 4)],
    ] {
        let n_rows = rows_of(&grids);
        let words = encode_vision_header(&grids, n_rows);
        assert_eq!(words.len(), 2 * grids.len() + 2);
        assert_eq!(words[0] as usize, grids.len());
        let (got_grids, got_rows) = decode_vision_header(&words).expect("round trip");
        assert_eq!(got_grids, grids);
        assert_eq!(got_rows, n_rows);
    }
}

#[test]
fn an_empty_payload_means_nothing_staged() {
    let words = encode_vision_header(&[], 0);
    assert_eq!(words, vec![0, 0]);
    let (grids, n_rows) = decode_vision_header(&words).expect("zero images is legal");
    assert_eq!(grids, Vec::<(usize, usize)>::new());
    assert_eq!(n_rows, 0);
}

#[test]
fn a_row_count_that_disagrees_with_the_grids_is_rejected() {
    let grids = [(2usize, 3usize), (4, 4)];
    let wrong = encode_vision_header(&grids, 23);
    let err = decode_vision_header(&wrong).expect_err("23 != 6 + 16");
    assert!(format!("{err:#}").contains("grids imply"), "{err:#}");
    // The honest count passes, so the assertion above cannot hold vacuously.
    assert!(decode_vision_header(&encode_vision_header(&grids, 6 + 16)).is_ok());
}

#[test]
fn a_truncated_or_empty_payload_is_rejected() {
    let words = encode_vision_header(&[(2, 3), (4, 4)], 22);
    for cut in 0..words.len() {
        assert!(
            decode_vision_header(&words[..cut]).is_err(),
            "truncation to {cut} words must be refused"
        );
    }
}

#[test]
fn a_zero_sized_grid_is_rejected() {
    let words = encode_vision_header(&[(2, 0)], 0);
    let err = decode_vision_header(&words).expect_err("an empty grid is malformed");
    assert!(format!("{err:#}").contains("empty grid"), "{err:#}");
}

#[test]
fn row_bases_are_the_prefix_sums_of_the_grids() {
    let grids = [(2usize, 3usize), (4, 4), (1, 5)];
    assert_eq!(stage_row_bases(&grids), vec![0, 6, 22]);
    assert_eq!(stage_row_bases(&[]), Vec::<usize>::new());
    // The last base plus its own rows is the total the header carries.
    let bases = stage_row_bases(&grids);
    assert_eq!(bases[2] + 5, rows_of(&grids));
}
