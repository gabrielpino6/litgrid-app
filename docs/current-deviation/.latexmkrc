# ======================================================
# latexmk configuration -- Enline (tex-enline template)
# ======================================================

$pdf_mode = 4;
$lualatex = 'lualatex -interaction=nonstopmode -shell-escape %O %S';

$bibtex_use = 2;

if ($^O eq 'MSWin32') {
    $pdf_previewer = 'start "" "%S"';
} elsif ($^O eq 'darwin') {
    $pdf_previewer = 'open -a Preview %S';
} else {
    $pdf_previewer = 'xdg-open %S';
}

$clean_ext = 'synctex.gz bbl blg run.xml';
