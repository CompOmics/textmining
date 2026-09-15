"""Merge prediction shards into run_meta_mlmarker_all_penalty.tsv and a
confidence>=0.3 filtered file mirroring Arnaud's two outputs."""
import glob, sys, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "/home/tine/git/Publication/textmining/final_analyses/MLMarkerPenalty/results"
parts = [pd.read_csv(f, sep="\t") for f in sorted(glob.glob(f"{R}/predictions_shards/shard_*.tsv"))]
df = pd.concat(parts, ignore_index=True).sort_values(["pxd", "run"]).reset_index(drop=True)
df.to_csv(f"{R}/run_meta_mlmarker_all_penalty.tsv", sep="\t", index=False)
df[df.confidence >= 0.3].to_csv(f"{R}/run_meta_mlmarker_penalty.tsv", sep="\t", index=False)
print("rows", len(df), "shards", len(parts))
print(df.scoring_mode.value_counts().to_dict())
print("top-1 share:", df.tissue.value_counts(normalize=True).head(6).round(3).to_dict())
print("Bone marrow top-1 by mode:", df.groupby("scoring_mode").apply(lambda g: round((g.tissue == "Bone marrow").mean(), 3)).to_dict())
