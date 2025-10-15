# Data

## Downloading the Data

I can't include the data files in the Git repo since it's 14.5GiB. There are currently three options to download the data, which are:

1. [Download from Zenodo](https://doi.org/10.5281/zenodo.17353963)
2. [Download from Proton Drive](https://drive.proton.me/urls/AVW808N4BM#nty2GuEUS4FR)
3. Using the BitTorrent file in this folder

Note: The file is a tar archive compressed with zstandard and will decompress to 44.5GiB.

## About using the Torrent
The torrent file is `pllm-ppi-data-leakage.torrent` in this folder, and you use it with a [torrent client](https://en.wikipedia.org/wiki/Comparison_of_BitTorrent_clients). [Transmission](https://transmissionbt.com/) is one such client available on Window, Linux, and MacOS and [a tutorial is available](https://www.mhaziqrk.uk/posts/2023/oct/how-to-install-transmission-and-how-to-use-torrents/). Alternatively, you can use [this link](https://webtor.io/a271ad4a07af1156d1ebea2e6066fc353e34dd34) to download it through your browser although it isn't recommended given the file's size. Once you download the file, you can help support us by seeding the file.


## What next?
After downloading, extract the file here using this command:

```
tar -I zstd -xvf data.tar.zstd
```

If you can't use `tar` on your platform, you can maybe use [PeaZip](https://peazip.github.io/index.html).

## Zenodo 
[![doi:10.5281/zenodo.17353963](https://zenodo.org/badge/DOI/10.5281/zenodo.17353963.svg)](https://doi.org/10.5281/zenodo.17353963)

Data has been deposited with Zenodo and accessible via the DOI: [10.5281/zenodo.17353963](https://doi.org/10.5281/zenodo.17353963).

If you use this data, please cite:

> Szymborski, Joseph, and Amin Emad. “Data for "A Flaw in Using Pre-trained pLMs in Protein-protein Interaction Inference Models"”. Zenodo, October 15, 2025. https://doi.org/10.5281/zenodo.17353963.

```
@dataset{szymborski_2025_17353963,
  author       = {Szymborski, Joseph and
                  Emad, Amin},
  title        = {Data for "A flaw in using pre-trained pLMs in
                   protein-protein interaction inference models"
                  },
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {10.5281/zenodo.17353963},
  url          = {https://doi.org/10.5281/zenodo.17353963},
}
```