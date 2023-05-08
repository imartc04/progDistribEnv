extern crate nalgebra as na;
use na::{DMatrix, DVector, U2};
// use std::cell::RefCell;
// use std::rc::Rc;
extern crate csv;

// use polars::prelude::LazyCsvReader;
use core::panic;
use ndarray::{arr2, Array2};
use ndarray_stats::CorrelationExt;
use polars::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

struct CBinGaussian {
    pub mean: DVector<f32>,
    pub cov: DMatrix<f32>,
    pub inv_cov: DMatrix<f32>,
}

impl Default for CBinGaussian {
    fn default() -> CBinGaussian {
        CBinGaussian {
            mean: DVector::from_vec(vec![0.0]),
            cov: DMatrix::from_vec(1, 1, vec![0.0]),
            inv_cov: DMatrix::from_vec(1, 1, vec![0.0]),
        }
    }
}

struct CBinPoisson {
    pub n: usize,
    pub N: usize,
    pub ln_n_N: f32,
}

impl Default for CBinPoisson {
    fn default() -> CBinPoisson {
        CBinPoisson {
            n: 0,
            N: 0,
            ln_n_N: 0.0,
        }
    }
}

struct CBinMultinomial {
    pub n: DVector<f32>, //Used as f32 because complicated problems with operations and types occurs if usize
    pub N: f32, //Used as f32 because complicated problems with operations and types occurs if usize
    pub ln_n_N: DVector<f32>,
}
impl Default for CBinMultinomial {
    fn default() -> CBinMultinomial {
        CBinMultinomial {
            n: DVector::from_vec(vec![0.0]),
            N: 0.0,
            ln_n_N: DVector::from_vec(vec![0.0]),
        }
    }
}

enum EBin {
    EGaussian(CBinGaussian),
    EPoisson(CBinPoisson),
    EMultinomial(CBinMultinomial),
}

impl Clone for CBinGaussian {
    fn clone(&self) -> Self {
        CBinGaussian {
            mean: self.mean.clone(),
            cov: self.cov.clone(),
            inv_cov: self.inv_cov.clone(),
        }
    }
}

impl Clone for CBinPoisson {
    fn clone(&self) -> Self {
        CBinPoisson {
            n: self.n.clone(),
            N: self.N,
            ln_n_N: self.ln_n_N,
        }
    }
}

impl Clone for CBinMultinomial {
    fn clone(&self) -> Self {
        CBinMultinomial {
            n: self.n.clone(),
            N: self.N,
            ln_n_N: self.ln_n_N.clone(),
        }
    }
}

impl Clone for EBin {
    fn clone(&self) -> Self {
        match self {
            EBin::EGaussian(g) => EBin::EGaussian(g.clone()),
            EBin::EPoisson(p) => EBin::EPoisson(p.clone()),
            EBin::EMultinomial(m) => EBin::EMultinomial(m.clone()),
        }
    }
}

struct CBin {
    pub id: usize,
    pub height: f32,

    //Option with the ids of the child CBins
    pub childs: Option<(usize, usize)>,
    pub dist_childs: f32,
    pub n_elements: usize,
    pub bin_spec: EBin,
    pub distances: HashMap<usize, f32>,
}

impl Default for CBin {
    fn default() -> Self {
        CBin {
            id: 0,
            height: 0.0,
            childs: None,
            dist_childs: 0.0,
            n_elements: 0,
            bin_spec: EBin::EGaussian(CBinGaussian::default()),
            distances: HashMap::new(),
        }
    }
}

impl Clone for CBin {
    fn clone(&self) -> CBin {
        CBin {
            id: self.id,
            height: self.height,
            childs: self.childs.clone(),
            dist_childs: self.dist_childs,
            n_elements: self.n_elements,
            bin_spec: self.bin_spec.clone(),
            distances: self.distances.clone(),
        }
    }
}

//Set distances between elements the passed vector of bins
fn set_bin_distances(mut f_bins: Vec<CBin>) -> Vec<CBin> {
    let num_bins = f_bins.len();
    (0..num_bins).for_each(|id1| {
        (id1 + 1..num_bins).for_each(|id2| {
            let dist: f32;
            match (&f_bins[id1].bin_spec, &f_bins[id2].bin_spec) {
                (EBin::EGaussian(gbin1), EBin::EGaussian(gbin2)) => {
                    dist = dist_gaussian_bins(gbin1, gbin2);
                }
                (EBin::EPoisson(pbin1), EBin::EPoisson(pbin2)) => {
                    dist = dist_poisson_bins(pbin1, pbin2);
                }
                (EBin::EMultinomial(mbin1), EBin::EMultinomial(mbin2)) => {
                    dist = dist_multinomial_bins(mbin1, mbin2);
                }
                _ => {
                    panic!("Specific bins must be of same type");
                }
            };

            let bin1_id = f_bins[id1].id;
            let bin2_id = f_bins[id2].id;
            f_bins[id1].distances.insert(bin2_id, dist);
            f_bins[id2].distances.insert(bin1_id, dist);
        });
    });

    f_bins
}

//The csv headers expected are  <record-id, feature1, ... , feature_n, Bin, class>
fn csv_to_gaussian_bins(f_csv_path: &String) -> Vec<CBin> {
    let q = CsvReader::from_path(f_csv_path)
        .unwrap()
        .has_header(true)
        .finish();
    let df = q.unwrap();

    //Vector of bins that will be returned
    let mut l_bins: Vec<CBin> = vec![];

    //Reference the id of the col Bin
    let l_bincol_id = df.find_idx_by_name("Bin").unwrap();

    //Obtain num of features
    let l_num_features = l_bincol_id - 1;

    //Obtain num of initial bins
    let max_id_bin = df.column("Bin").unwrap().max().unwrap();
    println!("Max bin id : {}", max_id_bin);
    //Define gaussian bin with known feature parameters sizes
    let l_gauss_bin = EBin::EGaussian(CBinGaussian {
        mean: DVector::from_vec(vec![0.0; l_num_features]),
        cov: DMatrix::from_vec(
            l_num_features,
            l_num_features,
            vec![0.0; l_num_features.pow(2)],
        ),
        inv_cov: DMatrix::from_vec(
            l_num_features,
            l_num_features,
            vec![0.0; l_num_features.pow(2)],
        ),
    });

    //Resize the vector as we will be accessing to it by id to set the data
    l_bins.resize(
        max_id_bin + 1,
        CBin {
            id: 0,
            height: 0.0,
            childs: None,
            dist_childs: 0.0,
            n_elements: 0,
            bin_spec: l_gauss_bin,
            distances: HashMap::new(),
        },
    );
    // println!("l_bins resized to : {} elements", l_bins.len());

    //Generate dataframe with num or elements and feature means for each Bin
    let df_mean = df
        .clone()
        .lazy()
        .groupby(["Bin"])
        .agg([count(), col("*").exclude(["Bin", "indice", "Class"]).mean()])
        .collect()
        .unwrap();

    println!("df_mean {}", df_mean);

    //Fill some values of the CBin's vector
    (0..df_mean.width()).for_each(|id_col| {
        df_mean
            .select_at_idx(id_col)
            .unwrap()
            .iter()
            .enumerate()
            .for_each(|(id_row, elem)| {
                //Obtain bin id of the row
                let val = df_mean.select_at_idx(0).unwrap().get(id_row).unwrap();
                let id_bin_row = match val {
                    AnyValue::Int64(x) => x as usize,
                    _ => panic!("Error in value"),
                };

                if id_bin_row > max_id_bin {
                    panic!("Not possible id larger than max");
                }
                let bin_row = &mut l_bins[id_bin_row];

                if id_col == 0 {
                    // println!("Setting id");
                    //Bin id column
                    bin_row.id = match elem {
                        AnyValue::Int64(val) => val as usize,
                        _ => panic!("Error in value id"),
                    };
                } else if id_col == 1
                //Num of elements column
                {
                    // println!("Setting n_elements");
                    bin_row.n_elements = match elem {
                        AnyValue::UInt32(val) => val as usize,
                        _ => panic!("Error in value n_elements"),
                    };
                } else {
                    //Feature colmn

                    // println!("Setting feature wit id : {}", id_col - 2);
                    match &mut bin_row.bin_spec {
                        EBin::EGaussian(gbin) => {
                            gbin.mean[id_col - 2] = match elem {
                                AnyValue::Float64(val) => val as f32,
                                _ => panic!("Error in value feature"),
                            };
                        }
                        _ => panic!("Error type distribution"),
                    };
                }
            })
    });

    // //Calcule covariances between features
    //For each bin cov(X,Y) = Σ [(Xi - X_mean) * (Yi - Y_mean)] / (n - 1)
    let df_only_features = df
        .clone()
        .lazy()
        .select(&[col("*").exclude(["indice", "Class"])])
        .collect()
        .unwrap();

    (0..l_bins.len() - 1).for_each(|bin_id| {
        let mask1 = df_only_features
            .column("Bin")
            .unwrap()
            .equal(bin_id as u64)
            .unwrap();

        let df_bin = df_only_features.filter(&mask1).unwrap();
        let df_bin_feat = df_bin
            .lazy()
            .select(&[col("*").exclude(["Bin"])])
            .collect()
            .unwrap();
        // println!("df_bin_feat {}", df_bin_feat);
        //Compute covariance values for each bin
        if df_bin_feat.height() > 0 {
            (0..df_bin_feat.width()).for_each(|id_feat1| {
                // println!("{:?}", df_bin_feat);
                // Convert the selected DataFrame to an Array2 for covariance calculation
                (id_feat1..df_bin_feat.width()).for_each(|id_feat2| {
                    //Make 2D Dataframe with features

                    let mut feat1_serie: polars::prelude::Series =
                        df_bin_feat.select_at_idx(id_feat1).unwrap().clone();

                    feat1_serie.rename("f1");

                    let df_cov = DataFrame::new(vec![
                        feat1_serie,
                        df_bin_feat.select_at_idx(id_feat2).unwrap().clone(),
                    ])
                    .unwrap();
                    // println!("df_cov_f1_f2 {}", df_cov);
                    let array = df_cov.to_ndarray::<Float32Type>().unwrap().reversed_axes();

                    // println!("Cov data array shape {}, {}", array.nrows(), array.ncols());
                    let cov = array.cov(1.0).unwrap();
                    //Get correspondent covariance value
                    let cov_value = cov[[0, 1]];

                    //Set covariance value in the bin
                    // println!("Array cov {}", cov);
                    match &mut l_bins[bin_id].bin_spec {
                        EBin::EGaussian(bing) => {
                            //In case id_feat1 == id_feat2 variance case, 2 assignations to the same value , no problem
                            bing.cov[(id_feat1, id_feat2)] = cov_value;
                            bing.cov[(id_feat2, id_feat1)] = cov_value;
                        }
                        _ => {
                            panic!("Only gaussian considered");
                        }
                    };
                });
            });

            //Once calculated bin cov calculate its inverse
            match &mut l_bins[bin_id].bin_spec {
                EBin::EGaussian(bing) => {
                    // println!("gaussian cov : {} , bin id : {}", bing.cov, bin_id);
                    //Set cov inverse
                    bing.inv_cov = bing.cov.clone().try_inverse().unwrap();
                }
                _ => {
                    panic!("Only gaussian considered");
                }
            }
        }
    });

    //Filter allocated bins with no elements
    let mut out_bins = Vec::new();
    l_bins.iter_mut().for_each(|bin| {
        if bin.n_elements > 0 {
            out_bins.push(bin.clone());
        }
    });

    // println!("Num elements out_bins : {}", out_bins.len());
    out_bins = set_bin_distances(out_bins);
    out_bins
}

fn dist_gaussian_bins(f_g1: &CBinGaussian, f_g2: &CBinGaussian) -> f32 {
    //Mean difference
    let l_diff = &f_g1.mean - &f_g2.mean;

    // Calculate the combined variance
    let mut comb_var = &f_g1.cov + &f_g2.cov;
    comb_var.try_inverse_mut();

    (l_diff.transpose() * comb_var * l_diff)[(0, 0)]
}

fn dist_poisson_bins(f_b1: &CBinPoisson, f_b2: &CBinPoisson) -> f32 {
    f_b1.n as f32 * f_b1.ln_n_N + f_b2.n as f32 * f_b2.ln_n_N
        - ((f_b1.n + f_b2.n) as f32) * (((f_b1.n + f_b2.n) / (f_b1.N + f_b2.N)) as f32).ln()
}

fn dist_multinomial_bins(f_b1: &CBinMultinomial, f_b2: &CBinMultinomial) -> f32 {
    *(&f_b1.n * &f_b1.ln_n_N + &f_b2.n * &f_b2.ln_n_N
        - (&f_b1.n + &f_b2.n) * ((&f_b1.n + &f_b2.n) / (&f_b1.N + &f_b2.N)).map(|x| x.ln()))
    .get((0, 0))
    .unwrap()
}

fn fusion_normal_bins(f_b1: &CBin, f_b2: &CBin, f_id: usize, f_distance: f32) -> CBin {
    let mut l_gauss_bin = CBinGaussian::default();
    match (&f_b1.bin_spec, &f_b2.bin_spec) {
        (EBin::EGaussian(gauss1), EBin::EGaussian(gauss2)) => {
            l_gauss_bin.inv_cov = &gauss1.inv_cov + &gauss2.inv_cov;
            l_gauss_bin.cov = l_gauss_bin.inv_cov.clone().try_inverse().unwrap();
            l_gauss_bin.mean = &l_gauss_bin.cov
                * (&gauss1.inv_cov * &gauss1.mean + &gauss2.inv_cov * &gauss2.mean);
        }
        _ => {
            // one or both values have different variants
        }
    }

    CBin {
        id: f_id,
        height: f_b1.height + f_b2.height + f_distance,
        childs: Some((f_b1.id, f_b2.id)),
        dist_childs: f_distance,
        n_elements: f_b1.n_elements + f_b2.n_elements,
        bin_spec: EBin::EGaussian(l_gauss_bin),
        distances: HashMap::new(),
    }
}

fn fusion_poisson_bins(f_b1: &CBin, f_b2: &CBin, f_id: usize, f_distance: f32) -> CBin {
    let mut l_bin = CBinPoisson::default();
    match (&f_b1.bin_spec, &f_b2.bin_spec) {
        (EBin::EPoisson(b1), EBin::EPoisson(b2)) => {
            l_bin.n = b1.n + b2.n;
            l_bin.N = b1.N + b2.N;
            l_bin.ln_n_N = ((l_bin.n / l_bin.N) as f32).ln();
        }
        _ => {
            // one or both values have different variants
        }
    }

    CBin {
        id: f_id,
        height: f_b1.height + f_b2.height + f_distance,
        childs: Some((f_b1.id, f_b2.id)),
        dist_childs: f_distance,
        n_elements: f_b1.n_elements + f_b2.n_elements,
        bin_spec: EBin::EPoisson(l_bin),
        distances: HashMap::new(),
    }
}

fn fusion_multinomial_bins(f_b1: &CBin, f_b2: &CBin, f_id: usize, f_distance: f32) -> CBin {
    let mut l_bin = CBinMultinomial::default();
    match (&f_b1.bin_spec, &f_b2.bin_spec) {
        (EBin::EMultinomial(b1), EBin::EMultinomial(b2)) => {
            l_bin.n = &b1.n + &b2.n;
            l_bin.N = l_bin.n.sum();
            l_bin.ln_n_N = (l_bin.n.clone() / l_bin.N.clone()).map(|x| x.ln());
        }
        _ => {
            // one or both values have different variants
        }
    }

    CBin {
        id: f_id,
        height: f_b1.height + f_b2.height + f_distance,
        childs: Some((f_b1.id, f_b2.id)),
        dist_childs: f_distance,
        n_elements: f_b1.n_elements + f_b2.n_elements,
        bin_spec: EBin::EMultinomial(l_bin),
        distances: HashMap::new(),
    }
}

struct COutData {
    //Use id_to_data hashmap to access children CBin from their ids stored in the parent
    pub head: usize,
    pub data: Vec<CBin>,
    pub id_to_data: HashMap<usize, usize>,
}

fn hierarchicalAlgo(mut f_bins: Vec<CBin>) -> COutData {
    //Obtain current max id
    let mut id = usize::MIN;
    f_bins.iter().map(|elem| {
        if elem.id > id {
            id = elem.id
        }
    });
    id = id + 1;
    // println!("Num elemns f_bins : {}", f_bins.len());
    //Generate auxiliar array of not merged bins
    let mut l_merge_bins: Vec<usize> = f_bins.iter().map(|bin| bin.id).collect();
    let mut bin_id_to_vec = HashMap::new();

    // println!("Num elements merge bins : {}", l_merge_bins.len());
    //Create direct association of bin id with the id of the input vector
    f_bins.iter().enumerate().for_each(|(id, bin)| {
        // println!("Inserting bin.id = {}, id_vec = {}", bin.id, id);
        bin_id_to_vec.insert(bin.id, id as usize);
    });

    //Reserve additional memory in f_bins for
    f_bins.reserve(f_bins.len() / 2);

    while l_merge_bins.len() > 1 {
        //Search for the unmerged pair of bins this least distance
        let mut bin_min_ids = [0, 0];
        let mut merge_bins_ids = [0, 0];
        let mut min_dist_val = std::f32::INFINITY;

        l_merge_bins.iter().for_each(|cbin_id| {
            // println!("Searching for id {}", cbin_id);
            f_bins[*bin_id_to_vec.get(&cbin_id).unwrap()]
                .distances
                .iter()
                .for_each(|(id_other_bin, dist_info)| {
                    // println!("Curr distance : {}", dist_info);
                    if dist_info < &min_dist_val {
                        // println!("Less than distance : {}", min_dist_val);
                        min_dist_val = *dist_info;
                        bin_min_ids[0] = *cbin_id;
                        bin_min_ids[1] = *id_other_bin;
                    }
                })
        });

        // println!("Min ids : {}, {}", bin_min_ids[0], bin_min_ids[1]);
        //Generate new candidate fusioning
        let bin1 = &f_bins[bin_id_to_vec[&bin_min_ids[0]]];
        let bin2 = &f_bins[bin_id_to_vec[&bin_min_ids[1]]];
        let mut new_bin: CBin;

        match (&bin1.bin_spec, &bin2.bin_spec) {
            (EBin::EGaussian(gauss1), EBin::EGaussian(gauss2)) => {
                new_bin = fusion_normal_bins(bin1, bin2, id, min_dist_val);
            }
            (EBin::EPoisson(pois1), EBin::EPoisson(pois2)) => {
                new_bin = fusion_poisson_bins(bin1, bin2, id, min_dist_val);
            }

            (EBin::EMultinomial(multi1), EBin::EMultinomial(multi2)) => {
                new_bin = fusion_multinomial_bins(bin1, bin2, id, min_dist_val);
            }
            _ => {
                // one or both values have different variants
                new_bin = CBin::default();
            }
        }

        //Delete merged bins from l_merge_bins
        l_merge_bins.retain(|&x| x != bin_min_ids[0] && x != bin_min_ids[1]);

        //Delete merged bins distances from remaining bins in l_merge_bins
        l_merge_bins.iter().for_each(|cbin_id| {
            f_bins[bin_id_to_vec[cbin_id]]
                .distances
                .retain(|&k, _| k != bin_min_ids[0] && k != bin_min_ids[1])
        });

        //Add new bin to the merge list
        l_merge_bins.push(new_bin.id);
        id = id + 1;

        //Calcule distances to new bin
        l_merge_bins.iter().for_each(|cbin_id| {
            let bin = &mut f_bins[bin_id_to_vec[&cbin_id]];

            let dist = match (&new_bin.bin_spec, &bin.bin_spec) {
                (EBin::EGaussian(gauss1), EBin::EGaussian(gauss2)) => {
                    dist_gaussian_bins(&gauss1, &gauss2)
                }
                (EBin::EPoisson(pois1), EBin::EPoisson(pois2)) => dist_poisson_bins(&pois1, &pois2),
                (EBin::EMultinomial(multi1), EBin::EMultinomial(multi2)) => {
                    dist_multinomial_bins(&multi1, &multi2)
                }
                _ =>
                // one or both values have different variants
                {
                    0.0
                }
            };

            bin.distances.insert(new_bin.id, dist);
        });

        //Add new bin to storage
        f_bins.push(new_bin);

        //Insert new bin into hash map
        bin_id_to_vec.insert(f_bins.last().unwrap().id, f_bins.len() - 1);
    }

    //Set CBin head
    let l_head_bin: &CBin;

    if l_merge_bins.len() == 1 {
        //The head element is last one remaining in the aux vector
        l_head_bin = &f_bins[bin_id_to_vec[&l_merge_bins[0]]];
    } else {
        //The head has to be the las pushed to data storage
        l_head_bin = f_bins.last().unwrap();
    }

    //Return algorithm dendogram
    COutData {
        head: l_head_bin.id,
        data: f_bins,
        id_to_data: bin_id_to_vec,
    }
}

fn main() {
    //Example read from csv and generate dendogram
    let bins = csv_to_gaussian_bins(&String::from(
        "/masterRSI/progEntDistr/entregaHierarAlgo/rust/hierarchical_clustering/data/datos.csv",
    ));

    let dendrogram = hierarchicalAlgo(bins);

    println!("Alogrithm finished");
}
