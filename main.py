import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto = "Hola. Bienvenidos a esta introducción a la inteligencia artificial generativa. Durante los próximos minutos vamos a descubrir cómo una computadora puede transformar texto en imágenes, sonidos y nuevas formas de contenido digital."

print("Generando audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7, # expresividad/emoción
    cfg_weight=0.4, # afecta seguimiento y ritmo. Si la voz va demasiado rápida 
    temperature=0.8, # controla variabilidad. Más alta = más variación, menos predecible
    repetition_penalty=1.2, # penaliza repeticiones extrañas
    min_p=0.05, # filtra opciones muy improbables durante la generación.
    top_p=1.0 # controla el muestreo probabilístico; 1.0 prácticamente lo deja abierto
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio guardado como audio.wav")
