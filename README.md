# Towards better Validity: Dispersion based Clustering for unsupervised Person Re-identification


Click [here](https://arxiv.org/pdf/1906.01308.pdf) to access the manuscript.

This code is based on the [Open-ReID](https://github.com/Cysu/open-reid) library and adopted from [BUC](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification).

## Preparation
### Dependencies
- Python 3.6
- PyTorch (version >= 0.4.1)
- h5py, scikit-learn, metric-learn, tqdm

### Download datasets 
- DukeMTMC-VideoReID: [[Direct Link]](http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip)  [[Google Drive]](https://drive.google.com/open?id=1Fdu5GK-C7P8M9QiLbiQNyT_RUFt8oFco)  [[BaiduYun]](https://pan.baidu.com/s/1qL39rnjTjyzjqaD-Wuv8KQ). This [page](https://github.com/Yu-Wu/DukeMTMC-VideoReID) contains more details and baseline code.
- MARS: [[Google Drive]](https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0)   [[BaiduYun]](https://pan.baidu.com/s/1mByTdvXFsmobXOXBEkIWFw).
- Market-1501: [[Direct Link]](http://45.32.69.75/share/market1501.tar)
- DukeMTMC-reID: [[Direct Link]](http://45.32.69.75/share/duke.tar)
- Move the downloaded zip files to `./data/` and unzip here.

## Usage

```shell
sh ./run.sh
```
`--size_penalty` parameter lambda to balance the intra-dispersion regularization term.

`--merge_percent` percent of data to merge at each iteration.






