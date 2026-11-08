
data/lambda_phage.fa
    Enterobacteria phage lambda, complete genome, RefSeq NC_001416.1 (48,502 bp).
    Test case for two-state GC-rich / AT-rich segmentation
    (Churchill 1989, Bull. Math. Biol. 51:79-94): the left ~22 kb is GC-rich.
    Obtained from the bowtie2 example data (example/reference/lambda_virus.fa).

data/chr21_train.fa, data/chr21_test.fa
    Real human sequence: 30 training and 8 test chunks of 10 kb each from
    chromosome 21q22.3 (GRCh38, chr21:43,020,001-43,670,000).
    Source: nf-core test-datasets copy of GRCh38 chr21.

data/chr21_train_labels.fa, data/chr21_test_labels.fa
    One record per chunk with the same header, one letter per base:
    H = inside a CpG island, L = background.
