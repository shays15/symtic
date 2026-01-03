# 📌 SyMTIC

## 🚀 Getting Started

This is the Github page assoicated with the paper "Synthetic multi-inversion time magnetic resonance images for visualization of subcortical structures".
We provide inference code and pretrained weights running SyMTIC.
If harmonization or imputation is required for out-of-distribution datasets, we recommend using HACA3 as shown in the paper.

Model weights can be downloaded here: https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/shays6_jh_edu/IQBIE87UIDmkTo3E_Dr6CC6mAVupjSDrLZVfgNCZWaXPS0o?e=1ChdsT

### 🧠 Prerequisites
- Python 3.10
- PyTorch ≥ 1.13
- See `requirements.txt` for full package versions

```bash
git clone https://github.com/shays15/symtic.git
cd symtic
pip install -r requirements.txt
```
## 🔍 Testing

To generate a multi-TI images:

```bash
python test.py \
  --input-mprage-path /path/to/mprage_image.nii.gz \
  --input-t2w-path /path/to/t2w_image.nii.gz \
  --input-flair-path /path/to/flair_image.nii.gz \
  --output-path /path/to/output.nii.gz
  --gpu 0
```

---

## 🧠 Citation

TODO
---

## 🙏 Acknowledgments

This material is partially supported by the **Johns Hopkins University Percy Pierre Fellowship** (Hays) and the **National Science Foundation Graduate Research Fellowship** under Grant No. **DGE-2139757** (Hays).
TODO
