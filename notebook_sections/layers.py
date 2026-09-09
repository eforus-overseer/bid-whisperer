"""Notebook section: the extra analysis layers. Exposes ``cells(md, code)``.

Assumes the notebook has already defined df, bidded, won, FIG, save, vs, dl, pf,
bl, plt, np, pd (see build_notebook.py). Chart forms follow the Visual Vocabulary
method: pick the relationship first, then the chart inside that family.
"""


def cells(md, code):
    md(r"""---
## More layers: what else the day tells us

The three assignment questions leave a lot of the log unread. This section adds
seven layers, each chosen for what a bidder could do with it the next morning: how
much we overpay when we win, how the day breathes hour by hour, how concentrated
the supply is, which browsers and operating systems click, whether cookie age
matters, what kinds of publishers hide behind the hashed domains, and where in the
country the clicks come from.

Each chart form was picked with the Visual Vocabulary method: name the
relationship the reader should see first (distribution, change over time,
part-to-whole, magnitude, correlation, spatial), then choose the chart inside that
family. The reasoning sits next to each figure. The code is in `src/layers.py`
and `src/layer_plots.py`.

One limit first. `user_id` is unique on every row: the file holds one impression
per user. Frequency capping, reach and ad fatigue cannot be studied in this
sample, and the check below is here so nobody tries.
""")
    code(r"""import layers as ly
import layer_plots as lp
import decode as dc

overall_ctr = float(won["conversion"].mean())
ly.user_uniqueness_report(df)""")

    # ------------------------------------------------------------------ bid shading
    md(r"""### Bid shading: paying for wins we already had

In a first-price auction the winner pays their own bid, and `feedback_bid` tells us
the runner-up. Everything between the two is money that did not change the
outcome. The relationship to show is a distribution (how is the overpay spread
across wins?), so the left panel is a histogram of overpay as a share of our bid.
The right panel replays the day with every bid cut by a flat percentage and tracks
two shares against the original: wins kept and spend saved.""")
    code(r"""stats = ly.overpay_stats(bidded)
fp_won = ly.overpay_frame(bidded)
curve = ly.shading_curve(bidded)

display(pd.Series(stats, name="value").to_frame().style.format("{:,.3f}"))
display(curve.loc[curve["shade"].isin([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])].style.format({
    "shade": "{:.0%}", "win_retained": "{:.1%}", "spend_saved": "{:.1%}",
    "cost_per_win_before": "{:.3f}", "cost_per_win_after": "{:.3f}", "impressions_lost": "{:,.0f}"}))
fig = lp.plot_overpay(fp_won, curve, stats); save(fig, "layer_overpay.png"); plt.show()""")
    md(r"""Half of our winning first-price bids sat more than 41% above the runner-up, and
in four wins out of ten we paid more than double what was needed. Summed over the
day, the runner-up prices come to 53% of what we paid, so the ceiling on savings is
about 47% of first-price spend. Reaching that ceiling needs the runner-up price
before the auction, which nobody has. A flat 10% shading is a policy we could
actually run: it keeps 88% of the wins, saves 18% of the spend, and cuts the cost
per won impression from 2.08 to 1.92 CPM. The losses tell the same story from the
other side. When we lose a first-price auction the winner is usually far above us
(median gap about 1 CPM) and only 4% of losses were within 10% of our bid. We are
rarely outbid by a hair; when we win, we win by too much.

A small data quirk: 273 won rows (0.7%) report a runner-up above our own bid. They
stay in the counts and are clipped to zero in the histogram.""")

    # ------------------------------------------------------------------ hourly
    md(r"""### The day's rhythm

Five quantities move through the same 24 hours, so this is change over time, and
the form is a column of small multiples on a shared hour axis rather than one
chart with five scales. The grey band marks 08 to 10 UTC, the middle of the night
for most of the United States.""")
    code(r"""hp = ly.hourly_profile(df, bidded, won)
display(hp.set_index("hour").T.style.format("{:,.3f}"))
fig = lp.plot_hourly(hp); save(fig, "layer_hourly.png"); plt.show()""")
    md(r"""Requests fall from about 16,000 an hour in the US afternoon to 3,200 at 09 UTC.
Our bid rate falls with them (56% at 01 UTC, 38% at 08 UTC), which suggests the
bidder filters harder when traffic is thin rather than chasing it. Win rate peaks
at 47% at 04 UTC and bottoms at 28% in the trough, while the median bid climbs from
0.57 CPM at midnight UTC to 0.93 CPM at 17 UTC: the afternoon is both busier and
more expensive per impression. CTR by hour swings between 7.6% and 12.0%, but each
hourly point rests on a few hundred to two thousand served impressions, so read
that line as a range rather than a schedule.""")

    # ------------------------------------------------------------------ concentration
    md(r"""### Volume is concentrated

The question is how the whole splits across publishers, so the family is
part-to-whole and the form is a Lorenz curve: domains ranked from smallest to
largest on the x axis, their cumulative share of impressions on the y axis. The
dashed diagonal is what equal shares would look like; the further a curve sags
below it, the more concentrated the supply. Gini summarises the sag in one number.""")
    code(r"""conc = ly.concentration_table(df)
display(conc.style.format({"gini": "{:.3f}", "top_10_share": "{:.1%}", "top_50_share": "{:.1%}",
                           "top_100_share": "{:.1%}", "top_500_share": "{:.1%}"}))
curves = {c: (ly.lorenz_points(df[c].value_counts().to_numpy()), ly.gini(df[c].value_counts().to_numpy()))
          for c in ("domain", "url", "ad_slot")}
fig = lp.plot_lorenz(curves, n_domains=int(df["domain"].nunique())); save(fig, "layer_lorenz.png"); plt.show()""")
    md(r"""The 50 largest domains (0.6% of 8,635) carry half of all impressions and the top
500 carry 87%. A Gini of 0.92 at the domain level is extreme even for web traffic.
For the bid model in question 3 this is good news: the pairs that matter most for
spend are exactly the ones with enough history to estimate a price curve, and the
long tail of one-off pages is where the pooled fallback earns its keep.""")

    # ------------------------------------------------------------------ browser / OS
    md(r"""### Browsers and operating systems

The user agent string is one of the few fields left in the clear. Comparing sizes
across a handful of categories is a magnitude story, so sorted horizontal bars do
the work: share of impressions on the left, CTR on served impressions on the right,
with thin categories (under 300 served) greyed out.""")
    code(r"""browser, os_table = ly.browser_os_summary(df, won)
fmt = {"share": "{:.1%}", "ctr": "{:.2%}", "n": "{:,.0f}", "served": "{:,.0f}"}
display(browser.style.format(fmt))
display(os_table.style.format(fmt))
fig = lp.plot_browser_os(browser, os_table, overall_ctr); save(fig, "layer_browser_os.png"); plt.show()""")
    md(r"""Chrome and Safari click at the same rate (9.9%), Edge trails at 8.1% and Firefox
at 5.7%. That fits the device story from question 2 more than it says anything
about browsers: Firefox and Edge are almost entirely desktop traffic. The operating
system split makes it plain. Android users click on 14.1% of served impressions,
iOS on 9.7%, Windows on 7.0%. For a CTR model, OS is a cleaner feature than
browser and largely a proxy for device type, so one of the two belongs in the
model, not both.""")

    # ------------------------------------------------------------------ cookie age
    md(r"""### Cookie age

The buckets are ordered, so this is change along an ordered axis and a line with
one point per bucket is the natural form. Support is printed under each bucket
because the buckets are very unequal in size.""")
    code(r"""ck = ly.cookie_age_ctr(won)
display(ck.style.format({"served": "{:,.0f}", "clicks": "{:,.0f}", "ctr": "{:.2%}"}))
fig = lp.plot_cookie_age(ck, overall_ctr); save(fig, "layer_cookie_age.png"); plt.show()""")
    md(r"""Half of all served impressions (23,232) go to a brand-new cookie, and those users
click at 9.0%, close to average. Cookies up to a day old click least (7.3%), those
up to a week old only slightly more (8.0%), and from there the rate climbs with
age to 10.5% for cookies older than a year. One reading: a new cookie is often a
first visit (curious, willing to click), while an established cookie belongs to a
regular whose behaviour is stable. Either way the shape is monotone after the
first day, which makes a bucketed cookie age a cheap and well-behaved CTR feature.
`cookie_age_seconds` takes only 77 distinct values, so it arrives pre-bucketed.""")

    # ------------------------------------------------------------------ archetypes
    md(r"""### Publisher archetypes

The hashes hide who a publisher is, but how its traffic behaves is visible: device
mix, language, viewability, session depth, auction type, connection, price level,
how often we win, how often users click, and the hour its traffic peaks. For the
294 domains with at least 100 impressions we standardise those traits and let
KMeans group them, choosing the number of groups by silhouette. Two traits plotted
against each other is a correlation story, so the chart is a bubble scatter:
device mix on the x axis, price level on the y axis, bubble area for volume, colour
for the group.""")
    code(r"""fp_dom = ly.domain_fingerprints(df, bidded, won, min_n=100)
arch = ly.publisher_archetypes(fp_dom)
print(f"{len(fp_dom)} domains with at least 100 impressions; k={arch['best_k']} chosen by silhouette")
display(arch["silhouette"].style.format({"silhouette": "{:.3f}"}))
cols = ["archetype", "n_domains", "impressions", "mobile_share", "second_price_share", "wifi_share",
        "viewability_mean", "session_depth_median", "median_bid", "win_rate", "ctr"]
display(arch["centroids"][cols].style.format({
    "impressions": "{:,.0f}", "mobile_share": "{:.0%}", "second_price_share": "{:.0%}", "wifi_share": "{:.0%}",
    "viewability_mean": "{:.0f}", "session_depth_median": "{:.1f}", "median_bid": "{:.2f}", "win_rate": "{:.0%}", "ctr": "{:.1%}"}))
print(f"desktop-only domains (under 5% mobile): {(fp_dom['mobile_share'] < 0.05).sum()}   "
      f"mobile-only (over 95%): {(fp_dom['mobile_share'] > 0.95).sum()}")
sp = df.loc[df["auction_type"] == "SECOND_PRICE", "domain"].value_counts()
print(f"second-price impressions: {sp.sum():,} across {len(sp):,} domains; the top 5 domains hold {sp.head(5).sum() / sp.sum():.0%} of them")
fig = lp.plot_archetypes(arch["labelled"], arch["names"]); save(fig, "layer_archetypes.png"); plt.show()""")
    md(r"""Three groups come out, and they are easy to recognise. About 180 domains are
desktop publishers: almost no mobile traffic, very deep sessions (a median near 28
impressions per session, which reads like long browsing sessions on content or
utility sites) and a low 6.9% CTR. About 95 domains are mobile publishers: 89% of
impressions on phones or tablets, short sessions, some cellular traffic and a
12.9% CTR. The third group is small (21 domains, 3% of impressions) and odd: 96%
of its auctions are second-price, the median bid is around 24 CPM, we win 88% of
the time, and its CTR is the highest of all at 14.2%. This is the premium
inventory that made `auction_type` look so strange in question 1, and it is
concentrated: five domains hold a quarter of all second-price impressions.

The clustering is soft (silhouette around 0.23), so the boundaries are fuzzy, but
for a bidder the three-way split is actionable: separate pacing and bid
strategies for desktop, mobile and the second-price premium pool, rather than one
global policy.""")

    # ------------------------------------------------------------------ state map
    md(r"""### Where the clicks are

Geography is a spatial story, and the fair form for state-level rates is a tile
grid map rather than a true map: every state gets the same square, so Rhode Island
is as readable as Texas and the eye is not drawn to land area, which has nothing
to do with impressions. Each tile prints the state code and its CTR; states with
fewer than 300 served impressions are hatched and left unrated, because a rate on
90 impressions is noise.""")
    code(r"""states = ly.state_summary(df, bidded, won)
display(states.head(15).style.format({"impressions": "{:,.0f}", "bid_rate": "{:.1%}", "bidded": "{:,.0f}", "win_rate": "{:.1%}",
                                       "median_bid": "{:.2f}", "served": "{:,.0f}", "clicks": "{:,.0f}", "ctr": "{:.2%}"}))
fig = lp.plot_state_tile_map(states); save(fig, "layer_state_map.png"); plt.show()""")
    md(r"""Among states with enough served impressions, Maryland (12.3%), Kentucky (12.0%),
Alabama (11.4%) and Iowa (11.0%) click most; Louisiana (6.5%), Utah (7.1%),
Massachusetts (7.2%) and Arkansas (7.5%) click least. The spread is wide, almost a
factor of two, but two cautions apply before treating geography as a strong
signal: state is missing on 3.4% of rows, and the mix of devices and publishers
differs by state, so part of this map is the device effect wearing a different
hat. That is consistent with the modest information value `state_code` earned in
question 2.""")

    md(r"""### What the extra layers add for a bidder

Two of them change money directly. Bid shading in first-price auctions is the
largest lever in the whole dataset: a flat 10% cut would have saved 18% of
first-price spend while keeping 88% of wins, and a per-slot shading policy built
on the question 3 price curves could approach the 47% ceiling. The hourly profile
says the afternoon is expensive and the early morning is cheap and easy to win,
which is an argument for time-of-day pacing. The rest sharpen the CTR feature set:
operating system (or device), cookie age and the publisher archetype are cheap,
stable and, in this sample, clearly separated.""")
