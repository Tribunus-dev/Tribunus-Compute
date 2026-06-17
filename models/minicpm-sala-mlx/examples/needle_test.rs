use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, is_stop_token, load_model, load_tokenizer, sample, strip_thinking,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(name = "needle_test", about = "Needle-in-a-haystack long context test")]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Target context length in tokens (e.g. 32000, 128000, 512000)
    #[arg(long, default_value_t = 32000)]
    context_len: usize,

    /// Needle insertion depth as fraction (0.0 = start, 1.0 = end)
    #[arg(long, default_value_t = 0.25)]
    depth: f32,

    /// Maximum tokens to generate for the answer
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
}

const NEEDLE: &str =
    "The secret verification code for Project Aurora is 739258. Remember this number carefully.";

const FILLER_PARAGRAPHS: &[&str] = &[
    "The annual rainfall in the Pacific Northwest varies between 150 and 200 centimeters depending on elevation and proximity to the coast. Meteorologists track these patterns using a network of ground stations and satellite imagery. The data collected helps farmers plan their crop rotations and irrigation schedules throughout the growing season.",
    "Modern bread baking techniques combine traditional fermentation methods with precise temperature control. The ideal proofing temperature for sourdough ranges from 24 to 27 degrees Celsius. Professional bakers monitor dough hydration levels carefully, as even small variations can significantly affect the final crumb structure and crust development.",
    "The history of lighthouse construction along the Atlantic seaboard spans three centuries of engineering innovation. Early wooden structures gave way to stone towers in the 1800s. Fresnel lens technology revolutionized the range of lighthouse beams, allowing ships to navigate dangerous coastlines from distances exceeding twenty nautical miles.",
    "Urban planning in European cities has evolved to prioritize pedestrian access and public transportation. Cities like Amsterdam and Copenhagen have invested heavily in bicycle infrastructure. Studies show that reducing car dependency improves air quality metrics by 15 to 30 percent while increasing retail activity in city centers.",
    "The migration patterns of monarch butterflies remain one of the most remarkable phenomena in the insect world. Each autumn, millions travel up to 4,800 kilometers from Canada to central Mexico. Researchers use tiny radio transmitters and isotope analysis to track individual butterflies across this incredible journey.",
    "Volcanic soil in regions like the Azores and Hawaii produces exceptionally fertile farmland. The high mineral content supports diverse agricultural output including coffee, pineapple, and various tropical fruits. Farmers on volcanic islands often maintain terraced fields that follow the natural contours of ancient lava flows.",
    "The development of fiber optic cables in the 1970s transformed global telecommunications. A single modern fiber can carry over 100 terabits per second across ocean floors. Submarine cable networks now span more than 1.3 million kilometers, forming the backbone of international internet connectivity.",
    "Traditional pottery techniques in East Asia involve multiple firing stages at temperatures exceeding 1200 degrees Celsius. Celadon glazes achieve their distinctive green hue through iron oxide reduction in oxygen-depleted kilns. Master potters spend decades perfecting the subtle variations that distinguish premium ceramics from ordinary ware.",
    "Antarctic research stations operate in extreme conditions with winter temperatures dropping below minus 60 degrees Celsius. Scientists stationed there study ice cores that contain atmospheric records spanning 800,000 years. These frozen archives provide invaluable data about historical climate patterns and greenhouse gas concentrations.",
    "The acoustic properties of concert halls depend on complex interactions between room geometry, surface materials, and air temperature. Engineers use computational fluid dynamics to model sound wave propagation. The reverberation time, measured in seconds, determines whether a hall is suited for orchestral music, chamber ensembles, or spoken word.",
    "Coral reef ecosystems support approximately 25 percent of all marine species despite covering less than 1 percent of the ocean floor. Reef-building corals depend on symbiotic algae called zooxanthellae for their energy. Rising ocean temperatures cause coral bleaching events that threaten biodiversity in tropical marine environments.",
    "The standardization of railroad gauge width in the 19th century was one of the most consequential infrastructure decisions in history. The 1435mm standard gauge, originally used by George Stephenson, became dominant across Europe and North America. Countries that adopted different gauges faced significant logistical challenges in cross-border freight transport.",
    "Archaeological discoveries at Gobekli Tepe in southeastern Turkey have fundamentally altered our understanding of Neolithic societies. The site features massive stone pillars carved with animal reliefs dating to approximately 9500 BCE. This predates agriculture and settled life, suggesting that monumental construction may have driven the transition to farming rather than the reverse.",
    "The chemistry of fermentation involves complex enzymatic pathways that convert sugars into ethanol and carbon dioxide. Saccharomyces cerevisiae, common baker's yeast, has been used for millennia in bread and beverage production. Modern genomic analysis has identified over 1,500 yeast strains with distinct fermentation characteristics used across different culinary traditions.",
    "Satellite imagery analysis reveals that global forest cover has declined by approximately 4.7 million hectares annually over the past two decades. Secondary growth forests, while valuable for carbon sequestration, support significantly less biodiversity than primary forests. Reforestation efforts must carefully consider native species composition to maximize ecological benefit.",
    "The physics of bridge design requires balancing tensile and compressive forces across span lengths ranging from meters to kilometers. Suspension bridges use high-strength steel cables that can support loads exceeding 200,000 tonnes. Modern computational modeling allows engineers to simulate wind load, seismic activity, and thermal expansion with unprecedented accuracy.",
    "Deep sea hydrothermal vents support unique ecosystems powered entirely by chemosynthesis rather than photosynthesis. Tube worms, clams, and specialized bacteria thrive at temperatures and pressures that would be lethal to most organisms. These extreme environments provide insights into the potential for life on other planets and moons in our solar system.",
    "The printing press, developed by Johannes Gutenberg around 1440, dramatically reduced the cost of book production. Before movable type, a single manuscript could take months to copy by hand. Within fifty years of its introduction, an estimated 20 million volumes had been printed across Europe, fundamentally transforming the spread of knowledge.",
    "Alpine glaciers serve as critical freshwater reservoirs for hundreds of millions of people downstream. The Rhone, Rhine, and Danube rivers all depend on glacial meltwater during summer months. Current retreat rates suggest that many smaller glaciers may disappear entirely within the next century, creating significant water security challenges.",
    "The human gut microbiome contains approximately 100 trillion microorganisms representing over 1,000 distinct species. Research has linked gut bacteria composition to conditions ranging from obesity to depression. Dietary fiber serves as a primary fuel source for beneficial microbes, producing short-chain fatty acids that support intestinal health.",
];

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("Loading model from {}...", args.model_dir);
    let load_start = Instant::now();

    let tokenizer = load_tokenizer(&args.model_dir)?;
    let model_args = get_model_args(&args.model_dir)?;
    let mut model = load_model(&args.model_dir)?;

    eprintln!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    // Build needle test prompt
    let question = "What is the secret verification code for Project Aurora? Answer concisely.";
    let needle_pos = (args.context_len as f32 * args.depth) as usize;

    // Insert needle at position
    let mut filler_tokens = Vec::new();
    let needle_tokens = tokenizer.encode(NEEDLE, false).unwrap().get_ids().to_vec();
    let question_tokens = tokenizer.encode(question, false).unwrap().get_ids().to_vec();
    let filler_parts: Vec<Vec<u32>> = FILLER_PARAGRAPHS.iter()
        .map(|p| tokenizer.encode(p, false).unwrap().get_ids().to_vec())
        .collect();

    // Build context up to needle_pos
    let mut total_tokens = 0usize;
    let mut context_tokens = Vec::new();

    'outer: loop {
        for part in &filler_parts {
            if total_tokens + part.len() >= needle_pos {
                // Insert needle here
                context_tokens.extend_from_slice(&needle_tokens);
                total_tokens += needle_tokens.len();
                // Continue with more filler
            }
            context_tokens.extend_from_slice(part);
            total_tokens += part.len();
            if total_tokens >= args.context_len {
                break 'outer;
            }
        }
    }

    // Truncate to exact context length
    context_tokens.truncate(args.context_len);
    context_tokens.extend_from_slice(&question_tokens);

    eprintln!("Context: {} tokens (needle at ~{}% depth)", context_tokens.len(), (args.depth * 100.0) as i32);
    eprintln!("Running needle test...");

    // Run inference
    let input_ids = mlx_rs::Array::from_slice(&context_tokens, &[1, context_tokens.len() as i32])?;
    let mut caches = create_layer_caches(&model.args);

    let prefill_start = Instant::now();
    let logits = model.forward(&input_ids, &mut caches)?;
    eval(&[&logits])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();
    let prefill_tok_s = context_tokens.len() as f32 / prefill_time;
    eprintln!("Prefill: {:.2}s ({:.1} tok/s)", prefill_time, prefill_tok_s);

    // Generate answer
    let mut last_token = sample(&logits, 0.0)?;
    let mut output_tokens = Vec::new();
    let decode_start = Instant::now();

    for _ in 0..args.max_tokens {
        if is_stop_token(last_token.item::<u32>()?) {
            break;
        }
        output_tokens.push(last_token.item::<u32>()?);

        let token_slice = last_token.reshape(&[1, 1])?;
        let logits = model.forward(&token_slice, &mut caches)?;
        eval(&[&logits])?;
        last_token = sample(&logits, 0.0)?;
    }

    let decode_time = decode_start.elapsed().as_secs_f32();
    let decode_tok_s = output_tokens.len() as f32 / decode_time;
    eprintln!("Decode: {:.2}s ({:.1} tok/s, {} tokens)", decode_time, decode_tok_s, output_tokens.len());

    let output_text = tokenizer.decode(&output_tokens, true).unwrap();
    println!("\nOutput: {}", output_text.trim());

    // Check if needle was found
    let found = output_text.contains("739258");
    eprintln!("\nNeedle found: {}", if found { "YES" } else { "NO" });

    Ok(())
}
