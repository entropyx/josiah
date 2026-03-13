import json
import os
import io
import zipfile
import pandas as pd


def export_scenario(df, ground_truth, path, fmt="csv", decomp_df=None):
    """Export a single scenario dataset + ground truth sidecar + optional decomposition.

    Args:
        df: DataFrame with scenario data.
        ground_truth: Dict of true parameters.
        path: Base path without extension.
        fmt: "csv" or "parquet".
        decomp_df: Optional decomposition DataFrame.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(f"{path}.parquet", index=False)
    else:
        df.to_csv(f"{path}.csv", index=False)

    with open(f"{path}_ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2, default=str)

    if decomp_df is not None:
        decomp_df.to_csv(f"{path}_decomposition.csv", index=False)


def export_batch_to_zip(results, fmt="csv"):
    """Export all scenarios to an in-memory ZIP file.

    Args:
        results: List of (DataFrame, ground_truth, decomp_df or None) tuples.
        fmt: "csv" or "parquet".

    Returns:
        BytesIO object containing the ZIP file.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, result in enumerate(results):
            df, gt = result[0], result[1]
            decomp = result[2] if len(result) > 2 else None
            name = gt.get("name", f"scenario_{i + 1:03d}")

            if fmt == "parquet":
                data_buf = io.BytesIO()
                df.to_parquet(data_buf, index=False)
                zf.writestr(f"{name}.parquet", data_buf.getvalue())
            else:
                csv_str = df.to_csv(index=False)
                zf.writestr(f"{name}.csv", csv_str)

            gt_str = json.dumps(gt, indent=2, default=str)
            zf.writestr(f"{name}_ground_truth.json", gt_str)

            if decomp is not None:
                decomp_str = decomp.to_csv(index=False)
                zf.writestr(f"{name}_decomposition.csv", decomp_str)

    buf.seek(0)
    return buf


def export_single_to_bytes(df, ground_truth, fmt="csv", decomp_df=None):
    """Export a single scenario to in-memory bytes.

    Args:
        df: DataFrame with scenario data.
        ground_truth: Dict of true parameters.
        fmt: "csv" or "parquet".
        decomp_df: Optional decomposition DataFrame.

    Returns:
        Tuple of (data_bytes, gt_bytes, data_filename, gt_filename,
                  decomp_bytes or None, decomp_filename or None).
    """
    name = ground_truth.get("name", "scenario")

    if fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        data_bytes = buf.getvalue()
        data_filename = f"{name}.parquet"
    else:
        data_bytes = df.to_csv(index=False).encode("utf-8")
        data_filename = f"{name}.csv"

    gt_bytes = json.dumps(ground_truth, indent=2, default=str).encode("utf-8")
    gt_filename = f"{name}_ground_truth.json"

    decomp_bytes = None
    decomp_filename = None
    if decomp_df is not None:
        decomp_bytes = decomp_df.to_csv(index=False).encode("utf-8")
        decomp_filename = f"{name}_decomposition.csv"

    return data_bytes, gt_bytes, data_filename, gt_filename, decomp_bytes, decomp_filename
