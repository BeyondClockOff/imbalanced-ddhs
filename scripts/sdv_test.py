from sdv.single_table import CTGANSynthesizer
from sdv.datasets.demo import download_demo


data, metadata = download_demo(
    modality="single_table", dataset_name="fake_hotel_guests"
)

synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=10)
synthetic_data
