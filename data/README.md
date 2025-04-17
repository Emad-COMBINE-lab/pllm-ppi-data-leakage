I can't include the data files in the Git repo since it's 16GiB, so I've created a torrent file which I'm seeding. The torrent file is `pllm-ppi-data-leakage.torrent` in this folder, and you use it with a torrent client. Alternatively, you can use [this link](https://webtor.io/9d437c5ab311ebe6dc761367a282e3b308e7551d) to download it through your browser although it isn't recommended given the file's size. Once you download the file, you can help support us by seeding the file.

You can download from there and extract the contents here. To extract the file, you can use this command:

```
tar -I zstd -xvf data.tar.zst
```

If you can't use `tar` on your platform, you can maybe use [PeaZip](https://peazip.github.io/index.html).