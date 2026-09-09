"""Notebook section: obfuscation forensics. Exposes ``cells(md, code)``.

Assumes df, bidded, won, save, plt, pd and display exist in the notebook. The
dictionary attack skips itself when the Tranco wordlist is absent, so the notebook
executes anywhere.
"""


def cells(md, code):
    md(r"""---
## Can the obfuscation be undone?

Five columns are hidden. `domain`, `url`, `ad_slot` and `publisher_properties`
arrive as 32-character hex strings and `user_id` is a UUID. Before trusting the
anonymisation we should test it the way a careful adversary would, cheap attacks
first. And even when the hashes hold, the relationships between them still
describe the hidden schema. Both questions get an answer below; the code is in
`src/decode.py`.""")
    code(r"""import decode as dc
import layer_plots as lp

fmt = dc.hash_format_report(df)
display(fmt.style.format({"share_32_lower_hex": "{:.1%}", "first_digit_max_share": "{:.3f}",
                          "first_digit_min_share": "{:.3f}", "n_distinct": "{:,.0f}"}))
nib = dc.nibble_frequencies(df)
fig = lp.plot_nibbles(nib); save(fig, "decode_nibbles.png"); plt.show()""")
    md(r"""All four fields are exactly 32 lowercase hex characters, the shape of an MD5
digest (or a truncated SHA), and the first digit of the distinct values is spread
evenly across 0 to f (between 5.6% and 6.9% each, against a uniform 6.25%). An
encoded or truncated identifier would show structure here; a digest does not.
`user_id` matches the UUID version 4 layout on every row, which means it is random
by construction and carries no information at all.

### What the structure leaks

If a hash is applied to a field one value at a time, equal inputs still give equal
outputs, so the joins between fields survive intact. The matrix below asks, for
each ordered pair, what share of the row field's values ever appear with more than
one value of the column field.""")
    code(r"""fd = dc.functional_dependency_matrix(df)
display(fd.style.format("{:.3f}"))
fig = lp.plot_fd_matrix(fd); save(fig, "decode_fd_matrix.png"); plt.show()""")
    md(r"""Three facts about the hidden schema fall out.

A page belongs to exactly one domain (`url` to `domain` is 1.000), so the `url`
hash is a hash of the full page and not of something that spans sites.

An ad slot is a site-level placement, not a page-level one. `ad_slot` pins down
the domain 98% of the time but the page only 79% of the time: one slot id (think
"leaderboard, top") is reused across many pages of the same site. That is why the
question 3 bid model pools by ad slot before it pools by domain.

`publisher_properties` sits above the domain. It fixes the domain only 87% of the
time, and a domain carries a single properties value only 79% of the time, so this
is a publisher-level configuration that several domains can share and that a
domain can switch during the day. It behaves like a seller or deal id rather than
a site attribute.

### Trying to reverse the hashes

Three cheap attacks cover what an attacker with a laptop would try first.
Dictionary: hash the one million most popular domains under nine common spellings
(bare, www, http, https, trailing slash, upper case) with four digest functions and
look for any match among the 8,635 domain hashes. Integers: hash the numbers 0 to
3,000,000, once and twice, because internal ids are often obfuscated that way.
Chaining: check whether one column is simply the hash of another. The dictionary
attack needs the Tranco list, which is not committed; when the file is absent the
cell says so and the ledger shows the result from the original run.""")
    code(r"""sets = dc.hash_sets(df)
display(dc.chained_hash_test(df))

ints = dc.integer_bruteforce(sets, n=1_000_000)
print(f"md5(str(i)) for i < {ints['n']:,}: {len(ints['hits'])} hits in {ints['seconds']:.1f}s")

live = [{"method": f"live: md5 of the integers 0 to {ints['n']:,}", "search_space": f"{ints['n']:,} digests", "hits": len(ints["hits"])}]
attack = dc.dictionary_attack(sets)
if attack["status"] == "ran":
    print(f"dictionary attack on {attack['wordlist']}: {attack['digests']:,} digests, {len(attack['hits'])} hits, {attack['seconds']:.0f}s")
    live.append({"method": f"live: {attack['n_words']:,} domains, 9 spellings, 4 digest functions",
                 "search_space": f"{attack['digests']:,} digests", "hits": len(attack["hits"])})
else:
    print("dictionary attack skipped:", attack["note"])
display(dc.attack_ledger(live))""")
    md(r"""Thirty-six million digests in the original run, and not one lands on a value in
the data. Neither does any integer or any chained hash. That is what a salted or
keyed hash looks like from the outside: the function is deterministic (the joins
survive) but the input space we can guess does not reach it. The obfuscation holds
against the cheap attacks, and there is no honest path to a domain name in this
file.

What remains is the side channel used in the archetypes section: a publisher's
behaviour (device mix, session depth, auction type, price level, peak hour) is in
the clear even when its name is not, and that is enough to tell a desktop content
site from a mobile app from a premium second-price deal. For the assignment that
is the useful conclusion. Identity is hidden and should stay hidden; structure and
behaviour are available and should be used.""")
