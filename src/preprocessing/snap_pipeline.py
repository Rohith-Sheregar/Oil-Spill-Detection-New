"""
SAR preprocessing via SNAP's `gpt` (Graph Processing Tool), driven by
subprocess -- NOT `snappy`. snappy's Python bridge is unreliable on recent
SNAP versions (version-locked jpy bindings, frequent segfaults on import,
and breakage across SNAP point releases). gpt via subprocess is slower to
iterate on interactively but far more stable for an unattended batch job.

Equivalent: pyroSAR (`pip install pyroSAR`) wraps this same gpt-graph
pattern with a nicer Python API. Worth trying first; fall back to this raw
XML-graph pattern if pyroSAR's assumptions don't fit your exact product.

Chain implemented (matches the synopsis exactly):
  Read -> ThermalNoiseRemoval -> Remove-GRD-Border-Noise -> Calibration
  -> Speckle-Filter (Refined Lee, 5x5) -> Terrain-Correction
  -> LinearToFromdB -> Write (GeoTIFF)
"""
import subprocess
from pathlib import Path

GRAPH_XML = """<graph id="preprocess">
  <node id="Read">
    <operator>Read</operator>
    <parameters><file>{input_path}</file></parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="Read"/></sources>
  </node>
  <node id="BorderNoise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources><sourceProduct refid="ThermalNoiseRemoval"/></sources>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct refid="BorderNoise"/></sources>
    <parameters>
      <outputSigmaBand>true</outputSigmaBand>
      <selectedPolarisations>VH,VV</selectedPolarisations>
    </parameters>
  </node>
  <node id="SpeckleFilter">
    <operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters>
      <filter>Refined Lee</filter>
      <filterSizeX>5</filterSizeX>
      <filterSizeY>5</filterSizeY>
    </parameters>
  </node>
  <node id="TerrainCorrection">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="SpeckleFilter"/></sources>
    <parameters>
      <demName>SRTM 3Sec</demName>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <mapProjection>AUTO:42001</mapProjection>
    </parameters>
  </node>
  <node id="ToDb">
    <operator>LinearToFromdB</operator>
    <sources><sourceProduct refid="TerrainCorrection"/></sources>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="ToDb"/></sources>
    <parameters>
      <file>{output_path}</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
</graph>"""


def run_preprocessing_graph(input_safe_path, output_path, gpt_executable="gpt", timeout=1800):
    """
    input_safe_path: path to the unzipped Sentinel-1 .SAFE product (gpt can
                      also read the .zip directly without unzipping first).
    Raises RuntimeError with SNAP's own stderr on failure -- SNAP's error
    messages are usually specific (e.g. naming the missing DEM tile or the
    exact heap-size error), so surface them rather than swallowing them.
    A crashed gpt process does not raise a clean Python exception on its
    own; this checks the return code explicitly to catch that case.
    On heap/OOM errors specifically: increase -Xmx in gpt.vmoptions
    (a sibling file next to the gpt executable) before retrying.
    """
    graph_path = Path(output_path).with_suffix(".graph.xml")
    graph_path.write_text(GRAPH_XML.format(input_path=input_safe_path, output_path=output_path))
    result = subprocess.run(
        [gpt_executable, str(graph_path)],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gpt failed (exit {result.returncode}):\n{result.stderr[-3000:]}")
    return output_path


if __name__ == "__main__":
    run_preprocessing_graph(
        "S1A_IW_GRDH_1SDV_20260315T000000_...SAFE",
        "scene_001_preprocessed.tif",
    )
