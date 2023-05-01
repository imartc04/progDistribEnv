extern crate nalgebra as na;
use na::{DMatrix, DVector};
use std::cell::RefCell;
use std::rc::Rc;
extern crate csv;

use polars::prelude::*;
use std::error::Error;
use std::fs::File;
use std::path::Path;

struct CBinGaussian {
    pub height: f32,
    pub mean: DVector<f32>,

    pub cov: DMatrix<f32>,
    pub inv_cov: DMatrix<f32>,
}

impl Default for CBinGaussian {
    fn default() -> CBinGaussian {
        CBinGaussian {
            height: 0.0,
            mean: DVector::from_vec(vec![0.0]),
            cov: DMatrix::from_vec(1, 1, vec![0.0]),
            inv_cov: DMatrix::from_vec(1, 1, vec![0.0]),
        }
    }
}

struct CBinPoisson {
    pub n: u64,
    pub N: u64,
    pub ln_n_N: f32,
}

struct CBinMultinomial {
    pub n: DVector<f32>,
    pub N: f32,
    pub ln_n_N: f32,
}

enum EBin {
    CBinGaussian,
    CBinPoisson,
    CBinMultinomial,
}

struct CBin {
    pub id: u64,
    pub height: f32,

    //Option with the ids of the child CBins
    pub childs: Option<(u64, u64)>,
    pub dist_childs: f32,
    pub n_elements: u64,
    pub bin_spec: EBin,
}

//The csv headers expected are  <record-id, feature1, ... , feature_n, bin, class>
fn csvToBins(f_csvPath: &String, f_vec: &mut Vec<CBin>) {
    // Path to the CSV file
    let path = Path::new(f_csvPath);

    // Open the CSV file
    let file = match File::open(path) {
        Err(e) => panic!("Failed to open file: {}", e),
        Ok(file) => file,
    };

    // Parse the CSV data to obtain num of initial bins
    let mut reader = csv::Reader::from_reader(file);
    let mut n_bins = 0;

    //Obtain bin id column and num of features
    // Get the first record, assuming it has headers
    let headers = reader.byte_headers()?;

    // Find the index of the field with the specified value
    let bin_id = headers.position(|header| header == "Bin");

    // Iterate over each record and print its fields
    for result in reader.records() {
        match result {
            Err(e) => println!("Error parsing CSV row: {}", e),
            Ok(record) => {
                n_bins = std::cmp::max(
                    n_bins,
                    record
                        .get(4)
                        .unwrap()
                        .clone()
                        .parse()
                        .expect("Failed to parse bin num"),
                );
            }
        }
    }

    if n_bins == 0 {
        panic!("No data inside csv file")
    }

    //Set vector size
    f_vec.resize(n_bins - 1);

    //Loop over thet records again to set bin's data
    for result in reader.records() {
        match result {
            Err(e) => println!("Error parsing CSV row: {}", e),
            Ok(record) => {
                //Fill CBin data
                let l_bin = f_vec.get_mut(record.get(4).unwrap());
            }
        }
    }
}

fn hierarchicalAlgo(f_csvPath: &String) {

    //Parse input to list of CBins
}

fn main() {}
