# Export Notes

When we build the ComfyUI graph, export the finalized workflow JSON into `workflow_templates/`.

Important:

- keep node ids stable after export
- do not hand-edit the graph topology in JSON unless strictly necessary
- prefer rebuilding and re-exporting the graph from ComfyUI if the topology changes
