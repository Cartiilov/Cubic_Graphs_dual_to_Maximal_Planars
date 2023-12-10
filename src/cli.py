import click
from planar_graph import PlanarTriangulation as plg
from PlotGraph import plot_graph, plot_distance_from_saved, plot_neighbours_from_saved, acquire_data_proc, acquire_data

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    """Planar Triangulation generation and analytics tool"""

@main.command()
@click.argument("num_of_nodes", type=int)
@click.argument("num_of_iters", type=int)
@click.argument("num_of_processes", type=int)
@click.option(
    "-p",
    "--plot-name",
    "pltname",
    type=str,
    required=False,
    help="Optional prefix to the name of the plots",
)
def get_data_with_proc(num_of_nodes, num_of_iters, num_of_processes, pltname):
    """Acquire data from a planar triangulation providing number of nodes, number of iterations and number of processes to run"""
    try:
        if pltname is None:
            acquire_data_proc(num_of_nodes, num_of_iters, num_of_processes)
        else:
            acquire_data_proc(num_of_nodes, num_of_iters, num_of_processes, pltname)

    except ValueError as e:
        click.echo("Ivalid")
        raise click.BadParameter("Invalid parameters") from e
    

@main.command()
@click.argument("num_of_nodes", type=int)
@click.argument("num_of_iters", type=int)
@click.option(
    "-p",
    "--plot-name",
    "pltname",
    type=str,
    required=False,
    help="Optional prefix to the name of the plots",
)
def get_data(num_of_nodes, num_of_iters, pltname):
    """Acquire data from a planar triangulation providing number of nodes, number of iterations"""
    try:
        if pltname is None:
            acquire_data(num_of_nodes, num_of_iters)
        else:
            acquire_data(num_of_nodes, num_of_iters, pltname)

    except ValueError as e:
        click.echo("Ivalid")
        raise click.BadParameter("Invalid parameters") from e
    

@main.command()
@click.argument("num_of_nodes", type=int)
@click.argument("save_name", type=str)
def generate(num_of_nodes, save_name):
    """Generate rendom planar triangulation and save to file providing number of it's nodes, name. Files providing graph structure can be found in 'data/graphs/'. Files with plots can be found in 'graph_plots/trian/"""
    try:
        plg1 = plg.random_planar_triangulation(num_of_nodes, save_name)
        plg1.sweep()
        plot_graph(plg1, save_name)
    except ValueError as e:
        click.echo("Invalid")
        raise click.BadParameter("Invalid parameters") from e

@main.command()
@click.argument("original_file_name", type=str)
@click.argument("destination_file_name", type=str)
def sweep(original_file_name, destination_file_name):
    """Sweep graph from provided file. Files providing graph structure can be found in 'data/graphs/'."""
    try:
        plg1 = plg.sweep_from_file(original_file_name, destination_file_name)
        plot_graph(plg1, destination_file_name)
    except ValueError as e:
        click.echo("Invalid")
        raise click.BadParameter("Invalid parameters") from e
    
@main.command()
@click.argument("original_file_name", type=str)
def find_cubic(original_file_name):
    """Find cubic graph which is dual to planar triangulation from your file. Files providing graph structure can be found in 'data/graphs/'."""
    try:
        plg1, t = plg.find_cubic_from_file(original_file_name)
        plot_graph(plg1, original_file_name, t)
    except ValueError as e:
        click.echo("Invalid")
        raise click.BadParameter("Invalid parameters") from e

