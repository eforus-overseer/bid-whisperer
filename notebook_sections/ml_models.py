"""Notebook section: machine-learning models (appended by build_notebook.py).

Exposes ``cells(md, code)``. The notebook's setup cell has already put ``src`` on
``sys.path`` and defined ``FIG``, ``save``, ``vs``, ``dl``, ``bl``, ``plt``; the Q1
cells defined ``df``, ``bidded`` and ``won``. Numbers quoted in the prose come from
``scripts/run_models.py`` with the default seeds, so a fresh execution reproduces
them.
"""


def cells(md, code):
    md(r"""---
## Beyond the brief: machine-learning models

The assignment asked for a feature analysis, not a model. Two models are still
worth fitting, for different reasons.

The first is a click-through-rate classifier. It checks whether the Q2 feature
ranking holds up when a model has to use those features together, and it puts a
number on the one thing the Q2 analysis could only gesture at: how much the hashed
identifiers (domain, url, ad slot, publisher properties) add once they are turned
into out-of-fold "historical CTR" encodings.

The second is quantile regression of the clearing price, trained and tested on
disjoint sets of url and ad-slot pairs. That is the cold-start case the Q3 lookup
model handles poorly: a slot it has never seen, where it can only fall back to the
domain or the global distribution.

Ground rules carry over from Q2. Models train on served impressions only, and no
auction outcome (bid, won_bid, feedback_bid) is a feature. All identifier encodings
are out-of-fold, so no row is scored with an encoding that saw its own label.
Code lives in `src/models.py`.

### A CTR classifier, with and without the identifiers
""")

    code(r"""import models as ml

ctr = ml.run_ctr_experiment(won)   # 5-fold stratified CV, fixed seed; about 15 s
display(ctr["metrics"].style.format("{:.4f}"))
print(f"Base click rate: {ctr['base_rate']:.2%} over {ctr['n_rows']:,} served impressions")
print(f"Lift from the identifier encodings: AUC {ctr['id_lift_auc']:+.3f}, average precision {ctr['id_lift_ap']:+.3f}")""")

    md(r"""Three models share the same folds. Gradient boosting on the context features
(device, state, hour, viewability, session depth, cookie age, wifi, auction type,
browser and OS parsed from the user agent) reaches an AUC of about 0.69. Logistic
regression on one-hot versions of the same inputs gets 0.68, so the booster is not
finding much non-linearity in the context alone. Adding the four out-of-fold
identifier encodings moves the booster to an AUC of 0.78 and raises average
precision from 0.17 to 0.26 against a base rate of 9.4%. The log loss drops from
0.289 to 0.266.

For a bidder that is the practical result of the whole Q2 exercise: the context
features are real but modest, and most of the predictable variation in clicks sits
in which page and which slot the ad runs on. The hashed IDs are not usable as raw
categories with tens of thousands of levels, but a smoothed historical click rate
per slot is cheap to maintain and is the single most valuable input.
""")

    code(r"""fig = ml.plot_ctr_roc_pr(ctr); save(fig, "ml_ctr_roc_pr.png"); plt.show()
fig = ml.plot_ctr_calibration(ctr); save(fig, "ml_ctr_calibration.png"); plt.show()""")

    md(r"""The reliability curve is the plot that matters for bidding. A CTR model is
used to price impressions (expected value = predicted CTR times value per click),
so a model that ranks well but is mis-calibrated still misprices. The decile bins
sit close to the diagonal across the range, and the predicted probabilities spread
from under 2% to over 30%, so the model separates cheap and expensive impressions
rather than hovering around the base rate.
""")

    code(r"""fig = ml.plot_ctr_importance(ctr, top=15); save(fig, "ml_ctr_importance.png"); plt.show()
display(ctr["importance"].head(10).style.format({"importance_mean": "{:.4f}", "importance_std": "{:.4f}"}))""")

    md(r"""Permutation importance (the drop in AUC when one column is shuffled on a
held-out fold) tells a slightly different story from the Q2 ranking, and the
difference is informative. Viewability comes first, at roughly 0.07 AUC, followed by
the ad-slot and url encodings and the publisher-properties encoding. Session depth
is next among the context features. Device brand, the top feature by information
value in Q2, drops well down the list.

That is not a contradiction. Device brand, device type and OS carry nearly the same
information (an Apple brand implies iOS implies a phone or tablet), and permutation
importance splits the credit across redundant columns, so each looks small on its
own. The Q2 ranking measured each feature alone; the model measures what each adds
given the others. Read together: keep one device signal, keep viewability and
session depth, and invest in the per-slot historical click rate.
""")

    md(r"""### Pricing slots we have never seen

The Q3 lookup model needs history for a slot. When the pair is new it falls back to
the domain, then to the auction-type global distribution. The question here is
whether a regression model that uses the request context can do better for new
pairs.

The test set is built by holding out 20% of url and ad-slot pairs, so every test
row is a pair the models never trained on. Only about 64% of test rows have a url
that appears anywhere in training, 59% an ad slot, and 87% a domain. Both models are
fitted on the same training rows. The regressor is gradient boosting with a pinball
loss on log clearing price at the 20th, 50th and 80th percentiles, with
auction-type-stratified encodings of domain, url and ad slot as extra inputs.
""")

    code(r"""qr = ml.run_quantile_experiment(bidded)   # grouped 80/20 split by (url, ad_slot); about 20 s
display(qr["table"].style.format({"quantile": "{:.1f}", "coverage": "{:.3f}", "pinball_cpm": "{:.3f}",
                                  "pinball_log": "{:.3f}", "median_bid_cpm": "{:.2f}", "coverage_gap": "{:+.3f}"}))
display(qr["by_auction_type"].style.format({"coverage_at_50": "{:.3f}"}))
print(f"Train pairs {qr['n_train_pairs']:,} | test pairs {qr['n_test_pairs']:,} | test rows {qr['n_test_rows']:,}")
print("Share of test rows whose key appears in training:", {k: round(v, 3) for k, v in qr['test_key_seen_in_train'].items()})""")

    code(r"""fig = ml.plot_qr_coverage(qr); save(fig, "ml_qr_coverage.png"); plt.show()
fig = ml.plot_qr_pinball(qr); save(fig, "ml_qr_pinball.png"); plt.show()""")

    md(r"""Coverage is the share of test clearing prices that fall below the recommended bid,
which is the win rate a bidder would actually get. Both models land within a few
points of their targets on slots they have never seen: the regressor at 0.28, 0.54
and 0.80 for targets of 0.2, 0.5 and 0.8, the lookup at 0.26, 0.52 and 0.80. Both
lean slightly high at the lower quantiles, which means bidding a little more than
needed to reach a low win rate. The lookup is marginally closer to target at 0.2 and
0.5.

Pinball loss is the proper score for a quantile forecast, and there the regressor
wins at every quantile: 0.961 against 0.968 CPM at the 20th percentile, 2.015
against 2.122 at the median, 2.468 against 2.495 at the 80th. The gap is small in
absolute CPM but it comes entirely from rows where the lookup has nothing specific
to say. Splitting the median by auction type shows the regressor is closer to target
on first-price traffic (0.55 against 0.52) and the lookup on the much smaller
second-price set (0.54 against 0.49).

What this means for a bidder: keep the lookup for slots with history, since it is
exact and needs no training, and route new pairs to the regressor instead of the
domain or global fallback. The two agree on the general price level (median
recommended bids of 1.61 and 1.31 CPM at the 50% target), so switching between them
does not produce jumps.

Caveats. This is one day of data, so there is no temporal holdout; a real deployment
would train on earlier days and test on later ones, and clearing prices drift. The
identifiers are hashed, so the models cannot generalise from the text of a url; the
encodings only work where the same hash recurs. And both experiments use the
feedback bid as the clearing price, which is a good but not exact market read on
lost first-price auctions.
""")
