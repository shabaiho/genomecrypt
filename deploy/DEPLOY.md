# Deployment

Two pieces, deployed separately.

| | Where | What it needs |
|---|---|---|
| Demo (Streamlit) | Hugging Face Spaces, Docker SDK | 2 GB image, AMRFinderPlus baked in |
| Landing (static) | Lovable | one self-contained HTML file |

The image was built and exercised end to end before this was written: inside the
container AMRFinderPlus runs, assembly QC passes at 5.43 Mb, the species check
confirms *K. pneumoniae* at 99.76% identity, and all five verdicts are produced.

---

## 1. Demo on Hugging Face Spaces

### Create the Space

1. https://huggingface.co/new-space
2. Owner: your account. Name: `genome-firewall`.
3. SDK: **Docker** → **Blank**. Hardware: **CPU basic** (free).
4. Visibility: Public.

### Push

```bash
git clone https://huggingface.co/spaces/<YOUR_HF_USER>/genome-firewall hf-space
cd hf-space

# everything the container needs, nothing else
rsync -a --exclude='.git' \
  ../genomecrypt/{src,app,config,models,pyproject.toml,uv.lock} .
mkdir -p data && rsync -a ../genomecrypt/data/demo data/
cp ../genomecrypt/deploy/Dockerfile .
cp ../genomecrypt/deploy/README-space.md README.md

git add -A
git commit -m "Genome Firewall"
git push
```

The first build takes 10–15 minutes: it installs BLAST+ and HMMER from Debian,
downloads the AMRFinderPlus binaries, and bakes in the ~230 MB AMR database.
Watch it under the Space's **Logs** tab.

### The URL

```
https://<YOUR_HF_USER>-genome-firewall.hf.space
```

That is the address to put in the landing page. The `huggingface.co/spaces/...`
URL also works but renders inside the Hub's chrome.

### Notes that matter

**The annotation version is pinned in the image.** The container ships
AMRFinderPlus 4.2.7 with database 2026-05-15.1, while local development uses
3.12.8 with 2024-07-22.1. The two extract slightly different features from the
same genome — measured on the demo genome: 23 determinants against 26, and the
newer database left *fewer* markers unrecognised (17% against 23%), so it is the
better default. Change it with `--build-arg AMR_VERSION=3.12.8` if you need the
deployed behaviour to match a local run exactly.

**Uploads do not survive a restart.** A Space's filesystem is ephemeral, so
`data/store/` — uploaded genomes, cached annotations, report history — is wiped
when the Space sleeps or rebuilds. The app degrades correctly: history shows
empty and every genome is annotated afresh. Attach persistent storage in the
Space settings if the history has to survive.

**Free Spaces sleep after inactivity.** The first visit after a sleep waits
through a cold start. For a scheduled demo, open it a few minutes early.

---

## 2. Landing on Lovable

The landing is one self-contained file: `landing/index.html`. No build step, no
external requests — the DNA helix is Canvas, the icons are inline SVG, the fonts
are system.

Before deploying, point the two demo buttons at the live Space:

```bash
./deploy/set-demo-url.sh https://<YOUR_HF_USER>-genome-firewall.hf.space
```

Then hand `landing/index.html` to Lovable.

---

## 3. Checks after deploying

```bash
SPACE=https://<YOUR_HF_USER>-genome-firewall.hf.space
curl -s -o /dev/null -w "%{http_code}\n" $SPACE/_stcore/health   # expect 200
```

Then, in the browser:

1. **Analyse example** — five verdicts, meropenem backed by `blaKPC`.
2. Upload `data/demo/test_files/wrong_species_K_oxytoca.fna` — every drug must
   come back without a verdict, organism reported as not confirmed at 89%.
3. Upload `data/demo/test_files/incomplete_assembly.fna` — refused at 4.05 Mb.

If the first check passes but the second returns verdicts, the species guard did
not run: check that `config/targets.fna` reached the image.
