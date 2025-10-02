from model_utils import *

def train_model_pipeline():
    generator, discriminator,train_loader, val_loader, test_loader = train_pix2pix("data/GAN-DATA/grid_dataset/grid_info.csv",
                                                                               epochs=100,
                                                                               batch_size=32,
                                                                               val_interval=1,
                                                                               save_dir="output/GAN_OUTPUT")
    
    plot_gan_history("output/GAN_OUTPUT/training_history.csv")
    
    # Assuming you already have your trained model and test_loader:
    visualize_pix2pix_results("output/GAN_OUTPUT/generator_final.pth", test_loader, num_samples=4)
    
    # Initialize the pipeline
    pipeline = GANInferencePipeline(
        generator_path="output/GAN_OUTPUT/generator_final.pth",  
        custom_font_path="fonts/RomanAntique.ttf", 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Example words to convert
    words = ["confieso", "los", "a","y", "noticias","mucho",'la','todo']

    # Generate handwriting
    input_grid, output_handwriting = pipeline.generate_handwriting(words)

    # Save results
    pipeline.save_results(input_grid, output_handwriting, 
                            output_dir="output/GAN_OUTPUT/inference_results", 
                            filename_prefix="sample")
    

if __name__ == "__main__":        
        train_model_pipeline()