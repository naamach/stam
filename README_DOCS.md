# Building the documentation

Local build (requires Python and pip):

1. Create and activate a virtual environment.
2. Install documentation requirements:

```bash
pip install -r docs/requirements.txt
```

3. Build HTML from `docs/`:

```bash
cd docs
make html
```

Read the Docs:

- Add the project to https://readthedocs.org and point it to this repository.
- RTD will use `.readthedocs.yml` and `docs/requirements.txt` to build the site.
