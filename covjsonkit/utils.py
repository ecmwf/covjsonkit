import os
from pathlib import Path


def merge_coverage_collections(collection1, collection2):
    """
    Merges two coverage collections into one.

    Args:
        collection1 (dict): The first coverage collection.
        collection2 (dict): The second coverage collection.

    Returns:
        dict: A merged coverage collection.
    """
    if not isinstance(collection1, dict) or not isinstance(collection2, dict):
        raise ValueError("Both collections must be dictionaries.")

    if collection1 == {}:
        return collection2
    if collection2 == {}:
        return collection1

    merged_collection = collection1.copy()

    if collection1.get("type") != collection2.get("type"):
        raise ValueError("Both coverageJSONs must be CoverageCollections.")
    if collection1.get("domainType") != collection2.get("domainType"):
        raise ValueError("Both coverageJSONs must be have the same domainType.")

    parameters1 = collection1.get("parameters", [])
    parameters2 = collection2.get("parameters", [])

    for parameter in parameters2.keys():
        if parameter not in parameters1.keys():
            merged_collection["parameters"][parameter] = parameters2[parameter]

    for coverage in collection2.get("coverages", []):
        merged = 0
        for coverage1 in merged_collection.get("coverages", []):
            if coverage["mars:metadata"] == coverage1["mars:metadata"]:
                if coverage["domain"]["axes"]["t"] == coverage1["domain"]["axes"]["t"]:
                    if (
                        coverage["domain"]["axes"]["composite"]["values"]
                        == coverage1["domain"]["axes"]["composite"]["values"]
                    ):
                        # Merge the data arrays
                        for param in coverage["ranges"].keys():
                            if param not in coverage1["ranges"].keys():
                                coverage1["ranges"][param] = coverage["ranges"][param]
                                merged = 1
        if merged == 0:
            # If no merge happened, append the new coverage
            merged_collection["coverages"].append(coverage)

    return merged_collection


def coverage_to_coveragecollection(coverage: dict) -> dict:
    """
    Wrap a single CoverageJSON object into a CoverageCollection.

    If the input is already a CoverageCollection, it is returned unchanged.

    Returns
    -------
    dict
        A CoverageCollection JSON structure containing the single coverage.
    """
    if not isinstance(coverage, dict):
        raise TypeError("Input must be a dict representing a CoverageJSON object")

    if coverage.get("type") == "CoverageCollection":
        # Already a collection, just return it
        return coverage

    if coverage.get("type") != "Coverage":
        raise ValueError("Input must be a Coverage object (type='Coverage')")

    collection = {
        "type": "CoverageCollection",
        "domainType": coverage.get("domain", {}).get("domainType", "Grid"),
        "coverages": [coverage],
        "referencing": coverage.get("referencing", []),
        "parameters": coverage.get("parameters", {}),
    }

    # include global metadata if present
    for key in ["title", "description", "attribution", "license"]:
        if key in coverage:
            collection[key] = coverage[key]

    return collection


def compress(in_path, out_path=None, compression=None, level=3):
    """Compress a file using the specified compression algorithm.
    Args:
        in_path (str): Path to the input file.
        out_path (str, optional): Path to the output file. If None, the output
            file will be created in the same directory as the input file with an
            appropriate suffix. Defaults to None.
        compression (str, optional): Type of compression to use. Options are 'zstd', 'LZ4'.
            Defaults to None.
        level (int, optional): Compression level. Defaults to 3.
    Returns:
        str: Path to the compressed file.
    """
    if compression is None:
        print("No compression specified, please specify type of compression (zstd, LZ4, binpack)")
    else:
        if compression == "zstd":
            import zstandard as zstd

            in_path = Path(in_path)
            in_file_size = os.path.getsize(in_path)

            if out_path is None:
                out_path = in_path
            output_file = in_path.with_suffix(out_path.suffix + ".zst")

            cctx = zstd.ZstdCompressor(level=level)

            with open(in_path, "rb") as f_in, open(output_file, "wb") as f_out:
                f_out.write(cctx.compress(f_in.read()))

            output_file_size = os.path.getsize(output_file)

            print(f"Compressed {in_path} → {output_file}")
            print(f"Input file size: {in_file_size} bytes")
            print(f"Output file size: {output_file_size} bytes")
            print(f"Compression ratio: {in_file_size / output_file_size:.2f}")

            return output_file

        elif compression == "LZ4":
            import lz4.frame

            input_file = Path(in_path)
            in_file_size = os.path.getsize(input_file)

            if out_path is None:
                output_file = input_file.with_suffix(input_file.suffix + ".lz4")
            else:
                output_file = Path(out_path)

            with open(input_file, "rb") as f_in:
                data = f_in.read()

            compressed = lz4.frame.compress(data, compression_level=level)

            with open(output_file, "wb") as f_out:
                f_out.write(compressed)

            output_file_size = os.path.getsize(output_file)

            print(f"Compressed {input_file} → {output_file}")
            print(f"Input file size: {in_file_size} bytes")
            print(f"Output file size: {output_file_size} bytes")
            print(f"Compression ratio: {in_file_size / output_file_size:.2f}")

            return output_file
    #    import binpacking

    #    compressed = binpacking.pack(data)
    #    return compressed
