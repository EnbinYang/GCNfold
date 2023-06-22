# GCNfold

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
make
sudo make install
```

Run `RNAplfold` to check whether the installation is successful. You can also find them in `/usr/local/bin`.
