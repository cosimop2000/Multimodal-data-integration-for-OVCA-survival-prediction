import pandas, sys
pandas.read_csv(
    sys.argv[1], header=0, sep="\t" if sys.argv[1].endswith(".tsv") else ","
).set_index("gene_id").transpose().to_csv(
    sys.argv[2], sep="\t" if sys.argv[2].endswith(".tsv") else ",", index_label="ID"
) if len(sys.argv) == 3 else print("You should pass as parameters the input and output files paths.")
