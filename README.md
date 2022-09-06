# nocturnalflightcalls.com

[![DOI](https://zenodo.org/badge/358788505.svg)](https://zenodo.org/badge/latestdoi/358788505)

This repository supports https://nocturnalflightcalls.com, a reference for bird nocturnal flight calls of North America.

## Organization
The main files for the webpage are listed in the following directories:
* `code/` - the code used to extract spectrograms and generate the tables for the website. See more information in the README inside of this directory.
* `css/` - stylesheets for the website
* `header/` - the pictures for the header
* `media/` - spectrograms and recordings for each species organized into folders by species. 
    * The `audio` subdirectory contains a maximum of 30 call recordings per species. To allow the ear's adjustment and to give context, each recording has a couple second buffer of background noise before and after the call as annotated.
    * The `spectrograms` subdirectory contains 3 spectrograms for each recording: a spectrogram of the full recording, and two spectrograms of the species's isolated call (one regular and one denoised)
* `tables` - contains three table resources for the site. One of regular call spectrograms (`spectrograms.html`), one of denoised call spectrograms (`spectrograms_denoised.html`), and a table of the species still needed for the project (`needed.html`).

