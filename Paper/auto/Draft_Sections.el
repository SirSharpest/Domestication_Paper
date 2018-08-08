(TeX-add-style-hook
 "Draft_Sections"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "twocolumn")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("geometry" "margin=0.6in") ("parskip" "parfill") ("hyphenat" "none")))
   (add-to-list 'LaTeX-verbatim-environments-local "minted")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "minted"
    "geometry"
    "fancyhdr"
    "lastpage"
    "float"
    "titlesec"
    "parskip"
    "subfig"
    "hyphenat"
    "framed")
   (LaTeX-add-labels
    "sec:orgf097e98"
    "sec:org5507ba5"
    "fig:org77286a5"
    "sec:org2052054"
    "sec:org2e61ab6"
    "fig:org23dd415"
    "sec:orgb162696"
    "fig:orgae230ab"
    "sec:org5093020"
    "sec:org8f214b3"
    "fig:org4801f4f"
    "sec:orge1ebce5"
    "fig:org6063a7c"
    "sec:orgaf07f43"
    "sec:orgdb9a6e0"
    "sec:org5f59b67"
    "sec:org466452f"
    "sec:org5959b54"
    "sec:org80fee3c"
    "sec:org1456d28"
    "fig:org913f1fd"
    "sec:org9656138"
    "eq:org5b42d70"
    "sec:org026fc5b"
    "sec:orge8f4a46"
    "eq:orgf431cba"
    "sec:org2d9677d"
    "sec:org2cc43ff"
    "sec:orga640f05"
    "sec:org8f2748d"
    "fig:org99df19c"
    "sec:org3729e91"
    "fig:org154c3db"
    "sec:orged0c1ed"
    "sec:org90888db"
    "sec:org225b393"
    "sec:org52c1b02"
    "sec:orgfeab4c0"
    "fig:org88202ae"
    "sec:org5520210"
    "fig:orgaf39bca"
    "sec:org3dfe9c1"
    "fig:org1a1c6bb"
    "sec:org00a1194"
    "fig:org0e66e7f"
    "sec:orge643028"
    "fig:org599ad24")
   (LaTeX-add-bibliographies
    "library"))
 :latex)

