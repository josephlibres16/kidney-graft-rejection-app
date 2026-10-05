# assets

The deployment assets go here. They are produced by `Export_Deployment_Assets.ipynb`
(run in Colab after the final model notebook) and unzipped into this folder:

- `final_tuned_xgboost.json`
- `probesets.txt.gz`
- `top20_dictionary.csv`
- `demo_specimens.npz`
- `deploy_meta.json`

The application refuses to start if any of them is missing.
