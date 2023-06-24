# GCNfold: A novel lightweight model for RNA secondary structure prediction

Let us spend some time configuring the environment that GCNfold needs.

### Install RNAplfold

`RNAplfold` is an executable file in the ViennaRNA package that generates the pairing probability of each base in the RNA sequence. We need to download the installation package from ViennaRNA website, tape archive the `.tar.gz` file, and set the permission of the parent directory.

```bash
mkdir rnafold
cd rnafold
wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_5_x/ViennaRNA-2.5.0.tar.gz
tar xvzf ViennaRNA-2.5.0.tar.gz
chmod 755 -R /root/rnafold  # change to your file path
```

We can start installing `RNAplfold` with the command below. Note that the `make` operation takes 20 minutes. The `make` process will appear many warnings. Just ignore them. This will not affect our use.

```bash
cd ViennaRNA-2.5.0
./configure
make
sudo make install
```

Run `RNAplfold`  to check whether the installation is successful. You can also find them in `/usr/local/bin`.

### Configure conda environment

We provide `environment.yml` files including all the environments that GCNfold depends on.

```bash
conda env create -f environment.yml
source activate rna_ss
pip install -e .  # install GCNfold module
```

Some additional packages need to be installed via the `pip` command. We used the <u>Tsinghua mirror source</u> when installing the dgl module. Not sure if this is possible outside of China. However, installing the 0.4.2 version of dgl by other ways should also be suitable.

```bash
pip install forgi
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/dgl-cu100/ package/dgl_cu100-0.4.2-cp37-cp37m-manylinux1_x86_64.whl
```

### Data Download

All data used in the experiments have been shared to [Google Drive](https://drive.google.com/drive/folders/1xfzHKbhYtOjyO9umKbHlUH1wAwdmUPRY?usp=sharing).

### ArchiveII dataset testing

ArchiveII is a dataset for extrapolation prediction. You can download it via [Google Drive](https://drive.google.com/drive/folders/1xfzHKbhYtOjyO9umKbHlUH1wAwdmUPRY?usp=sharing) link. Please put it in the folder `data/archiveII_all`. Run the command below to test. The performance results will be printed on the screen. RNA secondary structures will be stored in the `ct_file` folder in `.ct` file format. You can go through [RNApdbee](http://rnapdbee.cs.put.poznan.pl/) to visualize them.

```bash
python origin_test.py -c configs/archiveii_test.json
```

### DIY dataset testing

We also support you in building your own datasets. Please put your data (only `.ct` files are supported) in the `data/raw_data/diy_data` folder. We have stored 10 RNA data under this path in advance for your testing. Then go through the command below to package them into a `.pickle` file.

```bash
python preprocess_diy_data.py
```

Perhaps you need to remove duplicate data. This will generate a file called `test_no_redundant.pkl` under the `data/diy_data` file.

```bash
python filter_redundant_diy_data.py
```

Preparation is complete. Go ahead and test your DIY dataset below.

```bash
python diy_data_test.py -c configs/diy_data_test.json
```

### Paper Publication

This manuscript is currently <u>Under Review</u> in the journal *Computers in Biology and Medicine*.
