$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -shell-escape';
@generated_exts = (@generated_exts, 'synctex.gz');
@default_files = ('main.tex');
