# Pinned Overcooked probe reference

`overcooked_do_ablation_plots.py.txt` is the released analysis source at commit
`35a8430046531bd4b62e923d7507ee9f02fd97ea` of
https://github.com/ruaridhmon/emergent_partner_modelling.
Its SHA256 is `5e547d6788471624c3db36ea19eab9317e9ab372c2b844d5178c178fc64deefd`.
The upstream Apache license is preserved in `overcooked_LICENSE.txt`.

The strict grid analysis checks that hash and loads only the six probe
definitions via Python's AST. It does not execute the source's artifact loading
or plotting code. This reference is used by the current v1/v2 representation
analysis and its regression tests; it is not an old Overcooked experiment.
