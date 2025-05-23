# pllm-ppi-data-leakage
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI: 10.1101/2025.04.21.649858](https://badgen.net/badge/bioRxiv/10.1101%2F2025.04.21.649858/red)](https://doi.org/10.1101/2025.04.21.649858)


This repository houses the code and data for _"[A flaw in using pre-trained pLLMs in protein-protein interaction inference models](https://doi.org/10.1101/2025.04.21.649858)"_.

## How To Use this Repository

Please consult the [documentation](https://Emad-COMBINE-lab.github.io/pllm-ppi-data-leakage/
) for information on how to install dependencies, run experiments, and generate all figures in the manuscript.

## Where's the Data?

There are 16GiB of compressed data which is too large for Git to reasonably handle. Data files can be downloaded via the HTTP and BitTorrent protocol. More information is available in the [data](data/README.md) folder.

## Installation & Requirements

Instructions for installation, as well as details on requirements and dependencies, are all available through the [online documentation](https://Emad-COMBINE-lab.github.io/pllm-ppi-data-leakage/).

Install time for the various experiments in this repository vary greatly and depend on hardware (_e.g._, disk I/O), but typically will take less than 15 minutes for each code base.

Python 3 is required to locally building the online documentation site. Package dependencies is available in the `requirements.txt` file in this folder. All code was tested on Linux (Debian-based Distrubtions).

## Demo

A demonstration of how to regenerate all the figures in the manuscript is available through the [online documentation](https://Emad-COMBINE-lab.github.io/pllm-ppi-data-leakage/demo).

## License

### Code
All code files in this repository, unless otherwise specified, are licensed under the [GNU AGPLv3 License](https://www.gnu.org/licenses/agpl-3.0.html).

>    Code for "A flaw in using pre-trained pLLMs in protein-protein interaction inference models"
>
>    Copyright (C) 2025 Joseph Szymborski
>
>    This program is free software: you can redistribute it and/or modify
>    it under the terms of the GNU Affero General Public License as
>    published by the Free Software Foundation, either version 3 of the
>    License, or (at your option) any later version.
>
>    This program is distributed in the hope that it will be useful,
>    but WITHOUT ANY WARRANTY; without even the implied warranty of
>    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
>    GNU Affero General Public License for more details.
>
>    You should have received a copy of the GNU Affero General Public License
>    along with this program.  If not, see <https://www.gnu.org/licenses/>.

### Data
All data files in this repository, unless otherwise specified, are licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/Emad-COMBINE-lab/serious-flaw-pllm-ppi">Data for "A serious flaw in the design of pLLM-based protein-protein interactions"</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://jszym.com">Joseph Szymborski</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
