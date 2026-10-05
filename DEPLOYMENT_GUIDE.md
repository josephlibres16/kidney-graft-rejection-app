# Deployment guide

From the finished notebook to a public web link, using GitHub for the code and a free host for the
site. Budget about 45 minutes the first time.

Everything below is free. No credit card, no server to maintain.

## Choosing the host

The repository is ready for both. Step 3 covers Hugging Face Spaces, Step 3b covers Render.

| | Hugging Face Spaces (free) | Render (free) |
|---|---|---|
| Memory | 16 GB | 512 MB |
| CPU | 2 vCPU | under 1 CPU |
| Sleeps | after about 48 hours without visitors | after 15 minutes without visitors |
| Wake-up | around 30 seconds | around 1 minute |
| Monthly limit | none for CPU basic | 750 instance hours per workspace |
| Setup | upload files, Gradio is detected automatically | connect the repository, `render.yaml` is read automatically |
| Deploys from GitHub | through the included workflow | natively, on every push |

Both work. Spaces has far more headroom: a 310-biopsy upload is comfortable there, and the app
measured about 140 MB of memory for a 60-biopsy file during testing, so Render's 512 MB is workable
but not generous once Python, Gradio and XGBoost are loaded. `render.yaml` therefore caps uploads at
50 biopsies. The practical difference on defense day is the 15-minute sleep: on Render the site must
be opened a minute or two before the demonstration, every time.

---

## Step 1 — Export the model assets (Colab, once)

1. Open `Export_Deployment_Assets.ipynb` in Google Colab and mount Drive.
2. Check the three paths in the first code cell point at your Drive folders.
3. Run all cells.

The notebook re-reads the series matrix to recover the exact probeset order the model was trained on,
copies the saved model, picks eight held-out test biopsies as demos, and then **verifies** that the
exported model reproduces the scores recorded in the notebook run. If that check fails the notebook
stops; do not deploy a model that fails it.

Output: `deploy_assets.zip` (downloads automatically), containing

| File | What it is |
|---|---|
| `final_tuned_xgboost.json` | the trained model, exactly as saved by the analysis notebook |
| `probesets.txt.gz` | the 49,293 probeset IDs in the order the model expects |
| `top20_dictionary.csv` | top-20 data dictionary with training class averages |
| `demo_specimens.npz` | eight held-out biopsies for the built-in demo |
| `deploy_meta.json` | metrics, hyperparameters, versions and checksums for the About page |

---

## Step 2 — Put the project on GitHub

1. Create a free GitHub account if the team has none, then create a new **public** repository named
   `kidney-graft-rejection-app`. Do not add a README; this project already has one.
2. Unzip `deploy_assets.zip` into the `assets/` folder of this project, so `assets/` holds the five
   files listed above.
3. Upload the project. The simplest route needs no command line: on the empty repository page choose
   **uploading an existing file**, then drag in `app.py`, `requirements.txt`, `README.md`,
   `.gitignore`, `DEPLOYMENT_GUIDE.md` and the whole `assets/` folder, and commit.

   With Git installed, this is equivalent:

   ```bash
   cd kidney-graft-rejection-app
   git init
   git add .
   git commit -m "Kidney graft rejection predictor: Gradio app and model assets"
   git branch -M main
   git remote add origin https://github.com/<your-username>/kidney-graft-rejection-app.git
   git push -u origin main
   ```

All files are small, so Git LFS is not needed.

---

## Step 3 — Create the Hugging Face Space

1. Sign up at <https://huggingface.co> (free).
2. **New → Space.** Owner: your account. Space name: `kidney-graft-rejection`.
   SDK: **Gradio**. Hardware: **CPU basic (free)**. Visibility: **Public**.
3. On the new Space, open the **Files** tab → **Add file → Upload files**, and upload the same files
   you put on GitHub, keeping `assets/` as a folder. The Space rebuilds automatically.
4. Watch the **Logs** tab. The first build installs the dependencies and takes a few minutes. When it
   shows `Running`, the app is live at

   ```
   https://huggingface.co/spaces/<your-username>/kidney-graft-rejection
   ```

That URL is what you give your professor. It stays up permanently; a free Space goes to sleep after
about 48 hours without visitors and wakes itself on the next visit, taking roughly half a minute.

---

## Step 3b — Deploy on Render instead

Render builds straight from GitHub, so nothing is uploaded twice.

1. Sign up at <https://render.com> with the GitHub account that holds the repository.
2. **New → Blueprint**, choose `kidney-graft-rejection-app`, and confirm. Render reads `render.yaml`
   and creates a free web service with the right start command and settings. (Without a blueprint:
   **New → Web Service**, runtime **Python 3**, build command `pip install -r requirements.txt`,
   start command `python app.py`, instance type **Free**, and add the environment variable
   `MAX_SAMPLES = 50`.)
3. The first build takes a few minutes. When the service shows **Live**, the site is at

   ```
   https://kidney-graft-rejection.onrender.com
   ```

   The exact address appears at the top of the service page.
4. Every later push to `main` redeploys automatically.

Things to keep in mind on the free instance:

* **It sleeps after 15 minutes of no traffic** and takes about a minute to wake. Open the link a few
  minutes before the demonstration and keep the tab open.
* **512 MB of memory.** The app was built to stay well inside that: uploads are parsed in chunks
  straight into a 32-bit array rather than loaded whole. Keep `MAX_SAMPLES` at 50 and watch the
  **Metrics** tab after a large upload; if memory approaches the limit the service restarts and the
  page shows an error.
* **750 free instance hours per workspace per month**, which is enough for one service that sleeps
  between visits.
* `region: singapore` in `render.yaml` is the closest region to the Philippines. Change it if you
  prefer another.

---

## Step 4 (optional) — Keep GitHub and the Space in sync

So that pushing to GitHub updates the live site automatically:

1. On Hugging Face: **Settings → Access Tokens → New token**, role **write**. Copy it.
2. On GitHub: repository **Settings → Secrets and variables → Actions → New repository secret**,
   name `HF_TOKEN`, value the token. Add a second secret `HF_USERNAME` with your Hugging Face username.
3. Add the workflow file below as `.github/workflows/sync-to-hf.yml` and push.

```yaml
name: Sync to Hugging Face Space
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ secrets.HF_USERNAME }}
        run: |
          git push --force \
            https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/kidney-graft-rejection.git \
            main
```

Never commit the token itself; it belongs only in the repository secrets.

---

## Demonstrating it

1. Open the Space link in a full browser tab and press F11 for full screen.
2. **Demo biopsies** tab: pick a biopsy, press **Score this biopsy**. The card shows the prediction and
   the recorded diagnosis from GEO, which makes the demo verifiable in front of the panel.
3. **Score your own file** tab: upload a small CSV cut from the series matrix to show batch scoring and
   the downloadable results.
4. **About the model** tab: the held-out test numbers, the data dictionary and the proof-of-run details.

Have the Space open and awake a few minutes before the presentation so no one waits for a cold start.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Build fails with `Missing model assets in assets/` | The `assets/` folder was not uploaded, or the files sit at the repository root. They must be inside `assets/`. |
| Build fails while installing packages | Open the Logs tab and read the failing package. Loosening a version in `requirements.txt` usually fixes it. |
| App loads but every upload is rejected | The file does not use PrimeView probeset IDs such as `11719943_at`. Convert the file or use the series matrix directly. |
| Scores differ from the notebook | The exported assets are from a different run. Re-run the export notebook; its built-in check stops when the model does not reproduce the recorded scores. |
| Space or Render service is slow on the first visit | It was asleep. Open it once before the demonstration. |
| Render log says the service ran out of memory during an upload | The free instance has 512 MB. Lower `MAX_SAMPLES` (50 by default, try 20) or score fewer biopsies per file. |
| Render log says no open ports were detected | The app must listen on the port Render assigns. `app.py` already reads `PORT` and binds `0.0.0.0`; make sure the start command is `python app.py`. |
| Upload is rejected as too large | Hugging Face limits single files to 5 GB, but the browser upload is unreliable over a few hundred MB. Cut the series matrix down to the biopsies you want to score. |

---

## What to tell the panel about hosting

The model file, the probeset order and the data dictionary are version controlled on GitHub together
with the application code. The live site is built from that same repository, so the deployed model is
the one documented in the paper: `deploy_meta.json` carries the SHA-256 checksum of both the series
matrix and the model file, and the export step refuses to produce assets that do not reproduce the
recorded test scores.
