(TeX-add-style-hook
 "Draft_Sections"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("geometry" "margin=0.8in") ("parskip" "parfill")))
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
    "framed")
   (LaTeX-add-labels
    "sec:org700d30b"
    "sec:org3d5bf2d"
    "fig:org4a7a3b6"
    "sec:org6165991"
    "sec:orgd2c20ca"
    "sec:orgcc75f0c"
    "fig:org683801b"
    "fig:orge2c4b6a"
    "fig:orgba8a8b4"
    "fig:org77d5d95"
    "sec:org2468c2a"
    "sec:org495e4d9"
    "sec:orgfbbe8e8"
    "sec:org2f52b19"
    "sec:org3c0aa2e"
    "sec:org36dc96c"
    "sec:org3d92b5e"
    "fig:orgb4dd527"
    "sec:orga16b3bd"
    "eq:org5f9c827"
    "sec:org0d50ab6"
    "sec:orga500c80"
    "eq:orgc5f306d"
    "sec:org84dfb69"
    "sec:orgabcb0fd"
    "sec:orgf1d6972"
    "sec:org31bd52c"
    "fig:orge81711e"
    "sec:orgaa95095"
    "fig:org68e6e98"
    "sec:orgdf053ac"
    "sec:org2e15438"
    "sec:orgff21b2d"
    "sec:org3ba29ac"
    "sec:org936d935"
    "fig:org066959b"
    "sec:org2630b92"
    "fig:org810fb56"
    "sec:org5b46b88"
    "fig:orgfeee5a0"
    "sec:org93af86d"
    "fig:org0dbb402"
    "sec:org9992cc0"
    "fig:org7255b0d")
   (LaTeX-add-bibliographies
    "library"))
 :latex)

