# Data

## Downloading the Data
I can't include the data files in the Git repo since it's 16GiB. There are currently two options to download the data ahead of publication and its submission to the Zenodo data repository:

1. [Download from Proton Drive](https://drive.proton.me/urls/XP0QHQ51AM#d9HEKqgnvfEf)
2. Using the BitTorrent file in this folder

## About using the Torrent
The torrent file is `pllm-ppi-data-leakage.torrent` in this folder, and you use it with a [torrent client](https://en.wikipedia.org/wiki/Comparison_of_BitTorrent_clients). [Transmission](https://transmissionbt.com/) is one such client available on Window, Linux, and MacOS and [a tutorial is available](https://www.mhaziqrk.uk/posts/2023/oct/how-to-install-transmission-and-how-to-use-torrents/). Alternatively, you can use [this link](https://webtor.io/9d437c5ab311ebe6dc761367a282e3b308e7551d) to download it through your browser although it isn't recommended given the file's size. Once you download the file, you can help support us by seeding the file.


## What next?
After downloading, extract the file here using this command:

```
tar -I zstd -xvf data.tar.zstd
```

If you can't use `tar` on your platform, you can maybe use [PeaZip](https://peazip.github.io/index.html).