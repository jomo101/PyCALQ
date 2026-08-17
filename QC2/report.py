"""Summary-document builder for the QC2 Lüscher fit task.

Mirrors the document structure and method names of
``general.plotting_handler.PlottingHandler`` -- ``create_summary_doc`` ->
``append_section`` / ``append_subsection`` -> ``summary_table`` /
``add_single_plot`` / ``add_plot_series`` -> ``compile_pdf`` -- so QC2 output
reads as the same house style as the fvspectrum task reports.

It is deliberately standalone rather than a call into ``PlottingHandler``:
that module imports ``fvspectrum.sigmond_util`` and ``sigmond_scripts``, so
reusing it would make the report -- and therefore the whole task -- depend on
the correlator stack. QC2 starts from an HDF5 spectrum and otherwise needs
neither, and that independence is worth more here than sharing the code.

``pylatex`` and a LaTeX installation are required only to build the report; a
fit run without them still produces all plots, logs and results JSON.
"""

import logging
import os

try:
    import pylatex
    PYLATEX_AVAILABLE = True
except ImportError as exc:
    pylatex = None
    PYLATEX_AVAILABLE = False
    _PYLATEX_IMPORT_ERROR = exc


class FitReport:
    """Builds the single_channel_fit summary PDF."""

    def __init__(self, title):
        if not PYLATEX_AVAILABLE:
            raise ImportError(
                f"pylatex is required to build the fit report: {_PYLATEX_IMPORT_ERROR}. "
                "Install it with `pip install pylatex` (it is listed in requirements.txt), "
                "or set `report: false` in the task params to skip the summary document."
            )
        self.doc = None
        self.create_summary_doc(title)

    def create_summary_doc(self, title):
        """Create the LaTeX document and its preamble."""
        geometry = {"margin": "1in"}
        self.doc = pylatex.Document(geometry_options=geometry, page_numbers=True)
        self.doc.packages.append(pylatex.Package("float"))
        self.doc.packages.append(pylatex.Package("graphicx"))
        self.doc.packages.append(pylatex.Package("amsmath"))
        self.doc.preamble.append(pylatex.Command("title", title))
        self.doc.preamble.append(pylatex.Command("date", pylatex.NoEscape(r"\today")))
        self.doc.append(pylatex.NoEscape(r"\maketitle"))

    def append_section(self, title):
        self.doc.append(pylatex.Command("section", title))

    def append_subsection(self, title):
        self.doc.append(pylatex.Command("subsection", title))

    def add_paragraph(self, text):
        self.doc.append(pylatex.NoEscape(text))
        self.doc.append(pylatex.NoEscape("\n\n"))

    def summary_table(self, headers, data, title=""):
        """Add a centered table; headers and cells are passed through as LaTeX."""
        headers = [pylatex.NoEscape(str(h)) for h in headers]
        spec = "|".join(["c"] * len(headers))
        with self.doc.create(pylatex.Center()) as centered:
            if title:
                centered.append(pylatex.utils.bold(pylatex.NoEscape(title)))
                # Without an explicit paragraph break the title typesets on the
                # same line as the table body.
                centered.append(pylatex.NoEscape(r"\par\vspace{4pt}"))
            with centered.create(pylatex.Tabular(spec)) as table:
                table.add_hline()
                table.add_row(headers)
                table.add_hline()
                for line in data:
                    table.add_row([pylatex.NoEscape(str(col)) for col in line])
                table.add_hline()
        self.doc.append(pylatex.NoEscape("\n\n"))

    def add_single_plot(self, plotfile, caption=None):
        """Add one full-width figure."""
        if not os.path.exists(plotfile):
            logging.warning(f"Unable to include {plotfile} in summary pdf.")
            return
        with self.doc.create(pylatex.Figure(position="H")) as fig:
            fig.add_image(plotfile,
                          width=pylatex.NoEscape(r"\linewidth"),
                          placement=pylatex.NoEscape(r"\centering"))
            if caption:
                fig.add_caption(caption)

    def include_additional_plots(self, leftplotfile, rightplotfile):
        """Add up to two figures side by side."""
        left = os.path.exists(leftplotfile)
        right = rightplotfile is not None and os.path.exists(rightplotfile)
        if not left and not right:
            return

        if left and right:
            with self.doc.create(pylatex.Figure(position="H")):
                half = pylatex.NoEscape(r"0.5\linewidth")
                for path in (leftplotfile, rightplotfile):
                    with self.doc.create(pylatex.SubFigure(position="b", width=half)) as sub:
                        sub.add_image(path,
                                      width=pylatex.NoEscape(r"\linewidth"),
                                      placement=pylatex.NoEscape(r"\centering"))
        else:
            self.add_single_plot(leftplotfile if left else rightplotfile)

    def add_plot_series(self, files):
        """Add a series of plots, two per row."""
        for left, right in zip(files[::2], files[1::2]):
            self.include_additional_plots(left, right)
        if len(files) % 2:
            self.add_single_plot(files[-1])

    def compile_pdf(self, filename):
        """Compile to <filename>.pdf, falling back to writing the .tex source."""
        base = os.path.splitext(filename)[0]
        out_dir = os.path.dirname(os.path.abspath(base))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            self.doc.generate_pdf(base, clean_tex=False, silent=True)
            return f"{base}.pdf"
        except Exception as exc:
            # No LaTeX toolchain, or a compile error: keep the source so the
            # run still produces something useful.
            try:
                self.doc.generate_tex(base)
                logging.warning(
                    f"Could not compile the summary PDF ({exc}); wrote LaTeX source to {base}.tex"
                )
                return f"{base}.tex"
            except Exception as tex_exc:
                logging.warning(f"Could not write the summary document: {tex_exc}")
                return None
