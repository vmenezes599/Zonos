"""Test script for the TTS Client class."""

import random
from client import Client


def main():
    """Main function to test the TTS client with multiple requests."""
    # Initialize the client
    client = Client(host="127.0.0.1", port=9999)

    # Test connection first
    if not client.test_connection():
        print("Warning: Could not connect to server. Proceeding with requests anyway...")

    text_list = [
        """In the midst of a bustling harbor town, merchants shouted across wooden docks, sailors heaved ropes with weathered hands, and gulls circled above in hungry spirals. The scent of salt and tar mingled with roasted fish from nearby stalls. Children darted through alleys, chasing each other with laughter, while elders gathered near barrels sharing tales of distant voyages. Ships creaked as they swayed against their moorings, their sails patched with colors from faraway ports. At dusk, lanterns lit the cobblestones in warm amber hues, and music from fiddlers spilled into the night, binding the people of land and sea together.""",
        """Beneath the dense canopy of an ancient rainforest, the air shimmered with humidity and the soft hum of insects. Moss covered every fallen branch, and vines draped down like curtains hiding forgotten paths. A family of monkeys swung with practiced ease, screeching as they leapt between towering trees. Hidden birds burst forth in flashes of color, their wings scattering droplets of dew. The ground pulsed with life; ants carried burdens thrice their size, while frogs croaked unseen. Deep within, a waterfall thundered into a crystalline pool, a sacred place known only to a few, where legends whispered of spirits guarding the jungle.""",
        """On a windswept plateau, tall grasses bent and rippled like waves in an unseen ocean. The sky stretched endlessly, painted in shades of blue and silver. Nomads guided their herds across this vast plain, singing songs passed through generations, their voices rising against the gusts. Campfires flickered in the distance, carrying the scent of roasted meat and herbs. Children played with carved wooden toys while elders spun yarns of gods who rode storms. As twilight fell, the stars emerged in breathtaking clarity, entire galaxies unfurling above. In that serene, boundless silence, the plateau seemed a place where time itself paused.""",
        """The library was a cathedral of knowledge, its towering shelves packed with books bound in leather, cloth, and crumbling paper. Dust motes danced in golden shafts of sunlight piercing through stained glass windows. The scent of old parchment lingered in the air, comforting and ancient. Scholars whispered in hushed tones, scribbling notes onto parchment as they flipped fragile pages with reverence. A single candle burned at the desk of the caretaker, who watched with patient eyes as visitors came seeking wisdom. Beyond the shelves, secret doors whispered of hidden archives, where forgotten tomes waited for curious hands to rediscover them.""",
        """At the edge of the desert, where sand dunes rose like frozen waves, a small caravan paused at an oasis. Palms swayed gently, their fronds casting striped shadows over clear water. Traders unpacked spices, fabrics, and trinkets from across empires. Camels knelt with groans, relieved to sip cool water. The air smelled of cinnamon, cardamom, and sun-baked leather. Stories passed between travelers, tales of perilous journeys across endless dunes, of stars guiding them when all landmarks vanished. As night settled, the desert cooled, and the moon rose like a silver guardian, casting calm light over weary souls resting by the oasis.""",
        """The storm rolled across the coast with a force that shook every window and rattled every roof. Waves crashed against the cliffs, exploding into mist that drenched the land. In the village, shutters clattered while people huddled together around fires, listening to the roar outside. Lightning split the heavens in jagged lines, illuminating the terrified faces of fishermen who had not yet returned. Yet amidst the chaos, there was awe. Children peeked from blankets to watch the wild beauty of nature’s fury. When dawn came, the sea calmed, leaving behind wreckage but also the hope of rebuilding stronger than before.""",
        """The mountain path twisted upward, its stones worn smooth by centuries of footsteps. Pilgrims carried bundles of offerings, chanting as they ascended, their voices echoing in the thin air. Prayer flags fluttered, their colors vivid against the snow-capped peaks. Goats grazed along the slopes, indifferent to the sacred journey unfolding nearby. The wind carried both chill and whispers, as if the mountain itself spoke. At the summit stood a temple carved directly into stone, its doors weathered yet inviting. Inside, monks lit incense, the smoke curling upward like prayers made visible. The climb was grueling, yet every soul felt renewed.""",
        """In the heart of the old city, narrow streets twisted like veins, leading to squares where fountains bubbled and musicians strummed lutes. Stone buildings leaned with age, their balconies draped in bright fabrics and flowers. Market stalls spilled into the streets, filled with spices, olives, and handmade pottery. The air buzzed with bargaining voices, laughter, and the scent of roasting lamb. Children skipped with painted masks, mimicking traveling performers. A clocktower tolled the hour, its chimes mingling with the cacophony below. As dusk approached, lanterns illuminated the labyrinthine streets, turning the city into a mosaic of light, music, and endless life.""",
        """The farmhouse sat alone on a rolling hill, surrounded by fields that glowed golden in the late afternoon sun. The smell of freshly cut hay drifted in the breeze, mingling with the earthy scent of tilled soil. Chickens scratched near the wooden fence while a dog dozed lazily on the porch. Inside, bread baked in the oven, filling the rooms with warmth. Generations of photographs lined the mantel, each capturing stories of resilience. As night fell, fireflies rose like sparks from the grass, blinking in rhythmic patterns. Life was simple there, but rich in moments too often forgotten elsewhere.""",
        """The festival lit the night with fire and laughter. Torches lined the streets as dancers spun in vivid costumes, their faces painted with swirls of color. Drums thundered, guiding the rhythm of feet that stamped on cobblestones. Vendors sold sweets dipped in honey, drinks brewed from secret recipes, and charms carved from bone and silver. Children waved sparklers that drew bright trails in the dark, while elders watched from benches, smiling at the energy of youth. When midnight came, fireworks burst above, showering the crowd in brilliant cascades of light, a reminder that joy could burn brightly even in fleeting moments.""",
        """At the edge of the tundra, silence stretched for miles. Snow crunched beneath boots as explorers trudged forward, their breath forming clouds in the frigid air. Reindeer herds moved across the horizon, their antlers glinting under pale sunlight. In the distance, the aurora shimmered, ribbons of green and purple undulating like spirits dancing in the sky. Small fires fought against the cold, providing brief refuge. Tales of survival were etched in every line of the travelers’ faces, yet awe remained as strong as hunger. To stand in that vast expanse was to feel both fragile and immeasurably alive.""",
        """The ship drifted on calm seas, sails full under a gentle breeze. The deck creaked softly while sailors hummed songs to pass the time. Dolphins arced playfully beside the vessel, their sleek bodies glistening. Nets lay drying in the sun, smelling of salt and fish. A gull landed on the mast, cawing as if demanding attention. Below, in the galley, a pot simmered with stew, the aroma wafting upward. The horizon stretched endlessly, broken only by the promise of distant shores. Though storms and dangers awaited, in that moment, peace and freedom belonged to every soul aboard the ship.""",
        """The market at dawn stirred to life slowly, then all at once. Merchants unfolded tents, revealing fabrics dyed in impossible colors, jewelry that glittered, and foods that smelled of every spice under the sun. Farmers laid out baskets of figs, olives, and herbs. The first customers bargained quietly, but soon the air was filled with clamoring voices. Roosters crowed, dogs barked, and somewhere a flute played. Sunlight filtered between awnings, painting stripes of gold on the cobblestones. Children darted between stalls, clutching sweets. By midday, the square pulsed with life, a celebration of trade, community, and the simple joy of gathering.""",
        """High in a tower overlooking the city, a scholar bent over parchment illuminated by candlelight. Maps were strewn across the table, charts of stars and notes in careful ink. Outside, bells rang, echoing through quiet streets. The scholar paused to sip bitter tea, eyes weary but alight with discovery. Birds perched on the ledge, their feathers rustling as dawn approached. This tower had been home for decades, its walls filled with books and half-finished experiments. Each night of study was both a burden and a privilege, driven by a hunger to understand the mysteries that lay hidden in the cosmos.""",
    ]

    simple_text = "In the shadow of Khafre’s pyramid the Sphinx awakened to the dawn. Carved from a single ridge of limestone it faced the rising sun as guardian of rebirth and memory. Its gaze reached across centuries yet to come, its silence louder than prayer. Here stone became flesh and time took on a face. To stand before it was to meet the eternal watcher who measured both the reign of kings and the dust of ages."

    text_list = [simple_text]

    kwargs = {
        "happiness": 0.3077,
        "sadness": 0.0256,
        "disgust": 0.0256,
        "fear": 0.0256,
        "surprise": 0.0256,
        "anger": 0.0256,
        "other": 0.2564,
        "neutral": 0.3077,
        "expressiveness": 0.5,
        "speaking_rate": 0.375,
    }

    kwargs_2 = {
        "happiness": 0.6,
        "sadness": 0.0,
        "disgust": 0.4,
        "fear": 0.0,
        "surprise": 0.1,
        "anger": 0.0,
        "other": 0.1,
        "neutral": 0.2,
        "expressiveness": 0.5,
        "speaking_rate": 0.5,
    }

    kwargs_3 = {
        "happiness": 0.0,
        "sadness": 0.0,
        "disgust": 0.4,
        "fear": 0.0,
        "surprise": 0.1,
        "anger": 1.0,
        "other": 0.1,
        "neutral": 0.2,
        "expressiveness": 0.5,
        "speaking_rate": 0.1,
    }

    kwargs_4 = {
        "happiness": 0.1,
        "sadness": 0.1,
        "disgust": 0.4,
        "fear": 1.0,
        "surprise": 0.1,
        "anger": 1.0,
        "other": 0.1,
        "neutral": 0.2,
        "expressiveness": 0.5,
        "speaking_rate": 0.2,
    }

    # Create test requests
    test_requests = []
    for i, text in enumerate(text_list):
        test_request = {
            "text": text,
            "reference_audio_path": "/home/vitor/projects/DATABASES/CT_default_assets/voices/ElevenLabs_Clyde.mp3",
            "seed": random.randint(0, 2**32 - 1),
            "output_path": f"output/test_audio_{i}.mp3",
            # **kwargs,
            # **kwargs_2,
            # **kwargs_3,
            **kwargs_4,
        }
        test_requests.append(test_request)

    # Process each request
    for i, request in enumerate(test_requests):
        print(f"Processing request {i+1}/{len(test_requests)}")

        # Make the actual request
        success = client.synthesize(
            text=request["text"],
            reference_audio_path=request["reference_audio_path"],
            seed=request["seed"],
            output_path=request["output_path"],
            happiness=request["happiness"],
            sadness=request["sadness"],
            disgust=request["disgust"],
            fear=request["fear"],
            surprise=request["surprise"],
            anger=request["anger"],
            other=request["other"],
            neutral=request["neutral"],
            expressiveness=request["expressiveness"],
            speaking_rate=request["speaking_rate"],
        )

        if success:
            print(f"✓ Successfully generated audio for request {i+1}")
        else:
            print(f"✗ Failed to generate audio for request {i+1}")


if __name__ == "__main__":
    main()
