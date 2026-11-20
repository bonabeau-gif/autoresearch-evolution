# Generated research dashboard

The controller writes `progress.html` here after each completed generation. You can recreate it from `population.tsv` at any time:

`python -m autoresearch_evo dashboard`

The HTML is self-contained with no CDN dependencies. It is ignored by Git because research measurements remain machine-local until explicitly published. No example metric values are fabricated in the distributed archive.
