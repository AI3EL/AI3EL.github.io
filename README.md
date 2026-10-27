# AI3EL.github.io

Personal blog, served by GitHub Pages at <https://ai3el.github.io>.

## Writing

- Published posts live in `_posts/` as `YYYY-MM-DD-slug.md` (URL: `/blog/slug/`).
- Work in progress lives in `_drafts/` (not published).
- Math: `$$...$$` (inline inside a sentence, display when on its own lines).
- Figures:

  ```liquid
  {% include figure.html src="/assets/img/<post>/<file>.png" caption="..." width="80%" %}
  ```

## Local preview

```sh
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve --drafts   # http://localhost:4000
```

## Figures

Figure scripts are in `scripts/` and write to `assets/img/`. Run them with [uv](https://docs.astral.sh/uv/):

```sh
uv run scripts/q_learning_overfit.py   # real toy experiment, ~1 min
uv run scripts/make_diagrams.py        # schematic diagrams
```
