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
