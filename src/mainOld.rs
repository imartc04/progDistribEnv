extern crate nalgebra as na;
use na::{DMatrix, DVector};
use std::cell::RefCell;
use std::rc::Rc;

pub trait TBin<'a> {
    fn get_height(&self) -> f32;

    fn set_height(&mut self, f_height: f32);

    // fn dist_to(&self, f_other: &dyn TBin) -> f32;

    fn get_childs(&self)
        -> &Option<(Rc<RefCell<&'a dyn TBin<'a>>>, Rc<RefCell<&'a dyn TBin<'a>>>)>;

    fn set_childs(
        &mut self,
        f_childs: (Rc<RefCell<&'a dyn TBin<'a>>>, Rc<RefCell<&'a dyn TBin<'a>>>),
    );
}

struct CBaseBin<'a> {
    pub childs: Option<(Rc<RefCell<&'a dyn TBin<'a>>>, Rc<RefCell<&'a dyn TBin<'a>>>)>,
    pub dists: Vec<(Rc<RefCell<&'a dyn TBin<'a>>>, f32)>,
}

impl<'a> Default for CBaseBin<'a> {
    fn default() -> CBaseBin<'a> {
        CBaseBin {
            childs: None,
            dists: vec![0.0],
        }
    }
}

struct CBinGaussian<'a> {
    pub height: f32,
    pub mean: DVector<f32>,

    pub cov: DMatrix<f32>,
    pub inv_cov: DMatrix<f32>,
    pub base_bin: CBaseBin<'a>,
}

impl<'a> Default for CBinGaussian<'a> {
    fn default() -> CBinGaussian<'a> {
        CBinGaussian {
            height: 0.0,
            mean: DVector::from_vec(vec![0.0]),
            cov: DMatrix::from_vec(1, 1, vec![0.0]),
            inv_cov: DMatrix::from_vec(1, 1, vec![0.0]),
            base_bin: CBaseBin::default(),
        }
    }
}

impl<'a> TBin<'a> for CBinGaussian<'a> {
    fn get_height(&self) -> f32 {
        self.height
    }

    fn set_height(&mut self, f_height: f32) {
        self.height = f_height;
    }

    fn get_childs(
        &self,
    ) -> &Option<(Rc<RefCell<&'a dyn TBin<'a>>>, Rc<RefCell<&'a dyn TBin<'a>>>)> {
        &self.base_bin.childs
    }

    fn set_childs(
        &mut self,
        f_childs: (Rc<RefCell<&'a dyn TBin<'a>>>, Rc<RefCell<&'a dyn TBin<'a>>>),
    ) {
        self.base_bin.childs = Some(f_childs);
    }
}

fn dist_gaussian_bins(f_g1: &CBinGaussian, f_g2: &CBinGaussian) -> f32 {
    //Mean difference
    let l_diff = &f_g1.mean - &f_g2.mean;

    // Calculate the combined variance
    let mut comb_var = &f_g1.cov + &f_g2.cov;
    comb_var.try_inverse_mut();

    (l_diff.transpose() * comb_var * l_diff)[(0, 0)]
}


fn test_gauss(f_g : & CBinGaussian)
{
    
}

fn main() {
    let mut v : Vec<Rc<RefCell<dyn TBin>>>;
    v = vec![Rc::new(RefCell::new(CBinGaussian::default())), Rc::new(RefCell::new(CBinGaussian::default()))];
    // let mut v = vec![Rc::new(RefCell::new(CBinGaussian::default())), Rc::new(RefCell::new(CBinGaussian::default()))];

    test_gauss(*v[0].borrow())
}
