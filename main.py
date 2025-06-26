import click

@click.group()
def cli():
    """Arctic Sea Ice Data Fusion CLI"""
    pass

@cli.command()
@click.option('--input-file', default='assets/sea_ice_dataset_withoutint.mat', help='Input data file')
@click.option('--output-dir', default='cache/', help='Output directory for preprocessed data')
def preprocess(input_file, output_dir):
    """Preprocess raw data"""
    from arctic_sea_ice_data_fusion.preprocess import preprocess_data
    preprocess_data(input_file, output_dir)

@cli.command()
def train():
    """Train the model"""
    from arctic_sea_ice_data_fusion.train import train_model
    train_model()

@cli.command()
def evaluate():
    """Evaluate model performance"""
    from arctic_sea_ice_data_fusion.evaluate import evaluate_model
    evaluate_model()

@cli.command()
@click.option('--input', required=True, help='Path to input .mat file')
@click.option('--output', required=True, help='Path to output .mat file')
@click.option('--model', default='checkpoints/xgboost_model.pkl', help='Path to trained model file')
def inference(input, output, model):
    """Run inference on a .mat file and save results"""
    from arctic_sea_ice_data_fusion.inference import run_inference
    run_inference(input, output, model)

@cli.command()
@click.option('--data-file', default='assets/sea_ice_dataset_withoutint.mat', help='Sea ice data file')
@click.option('--save-plots/--no-save-plots', default=True, help='Save plots to files')
def visualize(data_file, save_plots):
    """Visualize sea ice data"""
    from arctic_sea_ice_data_fusion.visualize import visualize_sea_ice_overview
    visualize_sea_ice_overview(data_file, save_plots)

if __name__ == '__main__':
    cli()
