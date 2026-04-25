 
  #asmeconf: A latex template for ASME conference papers#
 
  Version 1.46 dated 2026/01/10.

  ####Overview####
  This class provides a LaTeX template for ASME Conference papers formatted according to
  the requirements on ASME's web pages (as posted in 2026):
  
  [www.asme.org/publications-submissions/proceedings/formatting-the-paper](https://www.asme.org/publications-submissions/proceedings/formatting-the-paper)
  
  A up-to-date BibTeX style for reference and citation formatting is also included.
  
  The asmeconf class is designed to approach the following aims:

- match ASME's font current specifications and layout

- match ASME's current reference formats, and add support for DOI and URL fields (asmeconf.bst replaces asmems4.bst)

- support hyperlinks to figures, tables, equations, references, and external URLs

- support pdf bookmarks and metadata

- set author names and addresses in either the traditional grid layout or the more recent inline style

- provide line numbers for editing and review

- support balancing of columns on last page

- support PDF/A standards (archival and accessible pdf formats)

- support copyright footer for federal employees and contractors

- enable various math and text font features

- support bold face, sans serif math in section headings

- support footnotes in section headings

- enable passages in other languages, e.g., for a translation of the abstract or a quotation

- support for many scripts: Latin, Arabic, Bengali, Chinese, Cyrillic, Devanagari, Greek, Hangul, Japanese, Tamil

  The .tex and .cls files are commented and should be self-explanatory.

  The files in this distribution are:

          README.md              --  this file
          asmeconf.cls           --  the class file
          asmeconf.bst           --  bibtex style for current ASME conference format
          asmeconf-sample.bib    --  a sample bibliography file
          asmeconf-style.css     --  css file for better rendering of tagged PDF
          asmeconf-template.tex  --  a latex template/example for this class
          asmeconf-template.pdf  --  documentation/sample paper
          *
          sample-figure-1.pdf, 
          sample-figure-2a.pdf, 
          sample-figure-2b.pdf   -- figures for the sample paper
          *
          examples/              -- a directory of examples: 
                                      -- nonlinear ode integration within LaTeX using Lua code
                                      -- tagged PDF/A-UA2 file that renders well as html
                                      -- grid-style layout of author names/addresses
                                      -- footers for government employees
                                      -- asmewide.sty and an example of use, for two-column equations
                                      -- fontspec and scripts for abstracts in 25 languages

  This work is not a publication of ASME itself. 
  
  ####Author####
  
  John H. Lienhard V
  
  Department of Mechanical Engineering
          
  Massachusetts Institute of Technology
          
  Cambridge, MA 02139-4307 USA


 ---
 
 ####Change log####
 v1.46 (2026/01/10)
 - For formats later than 2025/06/01, use native footnote support; fnpos and footmisc packages used for backward compatibility only
 - Rescale LeteSans and LeteSansBold math fonts to better match Heros text font
 - Various edits to css file to improve layout and to provide better cross-browser support
 - Uppercase "APPENDIX" and remove hanging indent from appendix titles when tagging is active (both had been lost in a previous edit)
 - Return pdf bookmarks to unnumbered sections (also lost in a previous edit)
 - Edit code related to captions to better capture various combinations of PDF/A, tagging, and LaTeX format date
 - Add: role/new-tag = { Paper\_number/Div }
 - Edit documentation
 
 v1.45 (2025/11/10)
 - Declare role/user-NS = asmeconf and change tag-add-new to role/new-tag.
 - Drop redundant Div.

 v1.44 (2025/11/08)
 - Class can now produce tagged PDF that validates as UA-2, A-4F, and WTPDF-1 using lualatex and no subcaptions 
 - Added example of UA-2 compliant PDF with attached css to the examples directory
 - Made various changes to support accessible pdf (UA-2):
  - included a css style file for better html rendering
  - replaced \\textit by \\emph in .bst for semantic emphasis and html rendering
  - in documentation, replaced \\textit by \\emph  for semantic emphasis and html rendering
  - unicode-math now loads bold math from the symbol font (\\mathbf=sym) for better html rendering
 - Fixed bug that caused hyperlinks in author affiliation numbers
 - Changed keywords format so that only “Keywords:” is in boldface sans-serif type
 - Fixed several malformed or undefined expl3 variables
 - Babel main language now defaults to USEnglish to match en-US in \\DocumentMetadata
 - Removed a spurious space in \\RequirePackage\[\\ac@TtwoA,\\ac@Tfive,\\ac@LGR,T1\]{fontenc}
 - Dropped default use of \[spanish\] option to avoid emergent interaction with \\@lbibitem
 - Edited implementation of caption fonts to accommodate 2025/11/01 \LaTeX release
 - Revised documentation
 - Regenerated example files

 v1.43 (2025/04/21)
 - Change loading order for sf greek glyphs
 - Other minor changes

 v1.42 (2025/04/17)
 - Rework the \[nofontspec\] class option, including font scaling and internal switches
 - Add \[loadscripts\] option for non-Latin scripts under \LuaLaTeX, to speed compilation. Simplify other font loading and add user-level \\if switches
 - Drop +ss03 (straight quote marks) for Inconsolatazi4 (typewriter font)
 - Fix bug in main language setting
 - Change fallback and documentation around former class option \[mathalfa=ccc=ddd\]
 - Edit class documentation and fontspec example file
 - Review and adjust backward compatibility; fix noncasechange in \\section
 - Miscellaneous code clean-up, including sectionnumberdepth in @startsec, encoding of operator font in sans and sansbold, set \\newif, etc.
 - PFD/UA-2 compliance (well-tagged PDF) is same as for v1.41
 
 v1.41 (2025/03/27)
 - \LuaLaTeX now loads unicode-math by default unless the class option \[nofontspec\] is used.
 - PDFA/UA-2 compliance is now possible using \LuaLaTeX by: omitting subcaptions and most language options, using compliant figures, omitting cross-references from captions, and loading the option \[captionpatch\], while using an appropriate invocation of \\DocumentMetadata.
 - Unicode-math will load Termes, Heros, Inconsolata, and Lete Sans Math fonts, with a few symbols drawn from STIX 2 fonts. Fira Math loads as a fallback to Lete Sans Math. These fonts are in TeX Live.
 - Eliminated legacy options \[pdfa, pdfaopt, pdfaconformance\]. PDF/A compliance is enabled by \\DocumentMetadata{..}
 - Eliminated \[mathalfa\] option. Instead, load mathalpha package in preamble or use unicode-math with range option.
 - Eliminated \[largesc\] option (now selected by default for pdftex).
 - Revised code for conference headers; added class option \[nohead\] to drop headers if desired.
 - Switch to \LaTeX's native key value mechanism, with the kvoptions package as a fallback.
 - Eliminated these packages: iftex, ifthen, hyphensubst. Eliminate these fallback packages: inputenc, hyperxmp, bookmark.
 - Load hologo package in the .tex file rather than the .cls file.
 - Fixed hanging indent of appendix section headings when tagging is active.
 - Fixed spurious vertical space after full line in abstract\* under lualatex.
 - Addressed color-related conflicts between luatexja and other fonts (removed \\Noto@color )
 - Rescaled Inconsolata font from 1.05 to 1.03 to better math lower case roman text
 - Ensured that english-language font is used for footer text under \LuaLaTeX when including other scripts.
 - Replaced \\int\_if\_zero:nTF by \\int\_compare:nTF for backward compatibility, and addressed related Expl3 compatibility issues.
 - Converted various code sections to Expl3 from \LaTeX2e, and removed legacy code here and there.
 - Removed legacy commands \\MakeTitlePage and \\PaperTitle (use \\maketitle and \\title)
 - Documented nomenclature options.
 - Fixed description list in Appendix B.
 - Made sample figures PDF/A-3u compliant.
 - Updated sample .bib file.
 - Update example files.
  
 v1.40 (2025/01/29)
 - Drop titlesec package to avoid degrading tag structure
 - Rework tagging of coffins in author/affiliation block
 - Fix minor bug that transferred spaces following \\affil and \\CorrespondingAuthor into superscripts

 v1.39 (2025/01/26)
 - Recode title and author blocks into expl3, adjust code to support tagged pdf
 - Revise other expl3 command and variable naming
 - Drop metalogo and xcoffins packages; use expl3 coffins directly.
 - Edit .bib and .tex files to replace \\LuaLaTeX by \\hologo{LuaLaTeX}
 - Move optional loading of \[textcase\] and \[inputenc\] to compatibility section
 - Edits to asmewide.sty: set \\@RuleWidth at end of \\maketitle; add \\ignorespacesafterend
 - Add an experimental and undocumented option \[captionpatch\] for tagging the caption package. With this option and pdftex, asmeconf can validate as PDF 2.0/a-4f provided that the \[spanish\] option is omitted. (UA-2 validation is not possible at present.)
 
 v1.38 (2024/08/01)
 - Add dcolumn type ",{x.y}" to center numbers on a comma separator
 - Change appendix section titles block format, rather than hanging indentation
 - Replace sample French appendix by a German appendix
 - Fix bug in \[main=\] option (for babel)
 - Define commands \\@title and \\@author 
 - Code clean-up (compatibility code, bookmark package loading, twocolumn loading,...)
 - Various code changes to better support future tagged pdf
 - Update documentation for using fontspec with asmeconf
 - Rescale Termes, Helvetica, and Inconsolata fontdimens under LuaLaTeX to match pdfTeX
 - Add logic to asme-template.tex to test for presence of non-latin fonts under LuaLaTeX
 - Change code for Japanese fonts to by-pass bug in luatexja-fontspec
 - Correct typo in babel font loading for traditional Chinese
 - Remove calls for Harfbuzz renderer since fontspec handles this internally
 
 v1.37 (2024/02/06)
 - Under LuaLaTeX, fontspec is now loaded by default. The option \[nofontspec\] will prevent this.
 - Eliminate the following long-deprecated options:  \[oldauthor, lang, lang-second, lang-third\]
 - newtxtext has removed the superiors font options (Jan 2024). As a result, the option \[nodefaultsups\] has been removed from asmeconf, along with supporting code
 - Streamline code for option handling and compatibility testing; other code clean-up
 - Switch code handling successive footnotes to expl3, adjust intersecting code
 - Small code changes to accommodate future tagged pdf
 - Refresh sample figures, edit sample bib file
 - Edits to asmeconf.bst: fix punctuation and spacing in journal title macros
 
 v1.36 (2023/12/28)
 - Change option-passing to xcolor package to address recent changes that package's code which lead to an error message.
 
 v1.35 (2023/10/19)
 - Adjust pdf-a code to accommodate recent changes to hyperxmp package. NOTE: \\DocumentMetadata{ ..} is now preferred to using class option [pdf-a].
 - Add option to not use boldface text in captions: \[unboldcaption\]
 - Add mathversion sans for sans-serif math and set as default in unbolded captions
 
 v1.34 (2022/12/30)
 - Roll back advanced metadata feature for compatibility with pre-2022 installations

 v1.33 (2022/12/30)
 - Update headers to 2023 IMECE
 - Add macro "jhmt" to asmeconf.bst for "ASME J. Heat Mass Transfer"; add sample citation and figures from jhmt.
 
 v1.32 (2022/09/14)
 - Incorporate patch to correct bug in hyperxmp that had impaired pdf/a compliance.
 - Add code to asmeconf.bst for empty page field in @article.

 v1.31 (2022/07/04)
 - Minor updates to address changes in the June 2022 release of LaTeX (PL 4) and the textcase package.
 - Add option to asmewide.sty to suppress final page column balancing, \[raggedend\], expand error message text.

 v1.30 (2022/03/14)
 - Edit code loading fonts for Greek, Vietnamese, and cyrillic languages under pdflatex, to ensure compatibility with newtx v1.71.  These options now require LaTeX distributions 2020/02/02 or later.
 - Edit font loading for the case when luaLaTeX is called without fontspec.
 
 v1.29 (2022/03/10)
 - Include current date and venue for 2022 IMECE (subject to change)
 - Edit example .tex file for asmewide.sty to remove description of unreleased development code
 - Fix bug in nofoot option that stopped capitalization of captions
 - Adjust scale factors on typewriter (raise to 1.05) and sans serif fonts (lower to 0.91)
  
 v1.28 (2022/02/14)
 - Introduce asmewide.sty, an experimental package for setting page-width equations in a two column format. A document with examples of use is included.
 - Increase scale of sans serif font under fontspec from 0.9 to 0.94, to better match newtxtext under pdflatex

 v1.27 (2021/12/26):
 - fix bug in captions that appeared in Jan. 2021 (code not uppercasing caption text)
 - revise code to switch to grid-style author block through a single package option, \[grid\]; previous macros for grid-style remain active. 
 - various code modifications to enable older LaTeX distributions to run this class. With pdfLaTeX, TeX Live 2016 or later should be used; with LuaLaTeX, TeX Live 2021 or later should be used.
 - eliminate use of \\entry{} with a single argument to produce subheadings in nomenclature; use \\EntryHeading{} instead. (**not backward compatible**)
 - incorporate recent changes to LaTeX pdf management in relation to pdf-a color profile loading and recently deprecated \\pdfcatalog command
 - minor edits in relation to url links
 - minor edits to asmeconf.bst
 - allow omission of affiliation superscript for single author or single institution
 - other minor code clean up
 
 v1.26 (2021/02/20):
 - edit title and metadata
 - put real author's name ahead of placeholder names

 v1.25 (2021/02/20):
 - edit descriptions
 - change \\mathsf in sansbold to be upright
 - move fontenc call under pdflatex options
 
 v1.24 (2021/01/26):
 - fix issue with math accents in headings & captions (Thanks to Beomjun Kye for reporting the problem)
 - adjust code for sans-serif upright Greek letters
 
 v1.23 (2021/01/18): 
 - Several minor edits and corrections
 
 v1.22 (2021/01/14): 
 - New command \\EntryHeading to produce subheadings in the nomenclature list;
 - New syntax for \\CorrespondingAuthor command with the authorgrid option (**not backward compatible**)
 - Add version field to the .bst file for publications online, manuals, and books;
 - Revise sample .bib file;
 - Enable language-specific fonts under pdflatex for Greek, Vietnamese, and cyrillic languages, with examples of abstracts in these languages;
 - Enable fontspec support for many scripts and languages under LuaLaTeX, including also Arabic, Bengali, Chinese, Japanese, Korean, and Tamil;
 - Provide example files for: authorgrid layout; integration of nonlinear ODE by LuaLaTeX; abstracts in 25 languages with fontspec+LuaLaTeX;
 - Extensively revise code that handles class options and keys, also add additional warning and error messages;
 - Create github issue tracker; 
 - Fix several minor bugs;
 - Edit documentation.

 v1.21 (2020/12/10): Edit documentation; update some usage of xparse under the hood; confirm compliance with substitutefont package. 
 
 v1.20 (2020/11/08): Add options to change copyright notice for federal employees and contractors; rename option "oldauthors" as "authorgrid"; other minor edits. Thanks to Bret Van Poppel for suggesting the federal copyright.
 
 v1.19 (2020/08/12): Add support for PDF/A archival standards (1b, 2b, 2u, 3b, 3u), as the newtx fonts have recently gained complete unicode maps; set pdf to load with single-page display rather than 2-page spread; add example of grid-style author block.
 
 v1.18 (2020/04/14): edit and expand documentation; revise sample .bib file; extensive edits to asmeconf.bst, to better support hyperlinks, to correct eid error, and for better conformance to ASME style (details listed in .bst file); add foreign language example.
 
 v1.17: set T1 font encoding with utf-8 input, ensure LuaLaTeX compatibility; load hologo and metalogo packages; edit documentation.
 
 v1.16: remove xpatch and comment packages from class file; disable \\( and \\) in pdf bookmarks to avoid warnings; edit documentation.
 
 v1.15: correct extra space left by \\CorrespondingAuthor when that author is not last; correct breakage of \\ref in captions.  Thanks to Bret Van Poppel for reporting these issues.
 
 v1.14: edit documentation; use 2020 IMECE header in layout example
 
 v1.13: add babel options for language support; minor text edits; adjust nomenclature list penalties
 
 v1.12: add support for line numbers for editing; add support for final column balancing; edit skips in nomenclature; adjust \\tolerance and \\emergencystretch (for line breaking); improve support for equation tags in captions; adopt standard \\maketitle and \\title commands; include \\versionfootnote for tracking revisions of draft.
  
 v1.11: minor adjustments to title, author, and affiliation layout
 
 v1.1:  revise several parts of the layout to match ASME's updated specifications from Summer 2019 (including author block, abstract font, placement of nomenclature, and minor spacings); add .bst support for online references and eprints; expand documentation significantly; guidance on fitting equations into columns. 
 
 v1.07: improve support for numbered section headings; allow omission of corresponding author email; edit documentation
 
 v1.06: automate bold sans math in captions and headings; small adjustments to default spacings; adjust font of paper number to 18 pt; edit documentation
 
 v1.05: minor code clean-up; change to keyvalue for to control font for superiors
 
 v1.04: fix option passing for mathalfa package; adjust \\entry to create nomenclature subheadings efficiently.
 
 
 ---
 
 ####License####

 Copyright (c) 2025 John H. Lienhard

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
 associated documentation files (the "Software"), to deal in the Software without restriction, 
 including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
 subject to the following two conditions:

 The above copyright notice and this permission notice shall be included in all copies or 
 substantial portions of the Software.

 The software is provided "as is", without warranty of any kind, express or implied, including but 
 not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. 
 In no event shall the authors or copyright holders be liable for any claim, damages or other liability, 
 whether in an action of contract, tort or otherwise, arising from, out of or in connection with the 
 software or the use or other dealings in the software.
