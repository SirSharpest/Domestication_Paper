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
    "sec:org1e43827"
    "sec:org48c2e01"
    "sec:org0b43068"
    "fig:org29a60ba"
    "sec:orgdb48674"
    "sec:org95473f6"
    "fig:orgec63b81"
    "sec:orgd9ffcf8"
    "fig:org69ae4ca"
    "sec:orgf51c491"
    "sec:org013bee9"
    "fig:org6f3821f"
    "sec:org7eb0037"
    "fig:orgb0cee6d"
    "sec:org81fa363"
    "sec:orgddebd21"
    "fig:orgb82ae62"
    "sec:org5c9345d"
    "sec:org835f426"
    "fig:orgdbcbc30"
    "sec:orgc1d55d7"
    "sec:org8b605ea"
    "sec:org081eb15"
    "sec:org9aa63da"
    "sec:orgbfaa441"
    "sec:org6e3c7eb"
    "sec:org846ffcf"
    "fig:org4071500"
    "sec:orgb71a6af"
    "eq:org7dd3ee8"
    "sec:orgd69cf0f"
    "sec:org8403576"
    "eq:org82e35bb"
    "sec:orgc4a97b1"
    "sec:org7ccd31c"
    "sec:orgc786e53"
    "sec:org26b3f81"
    "fig:orga51c863"
    "sec:orgdee0062"
    "sec:org29d101a"
    "sec:orgeaf88c5"
    "fig:org62f1159"
    "sec:org7266207"
    "fig:org60d0038"
    "sec:orgf491518"
    "fig:org6fcf241"
    "sec:org55f6d89"
    "fig:org778d7cd"
    "sec:org6a1967a"
    "fig:org11bf147")
   (LaTeX-add-bibliographies
    "library"))
 :latex)

