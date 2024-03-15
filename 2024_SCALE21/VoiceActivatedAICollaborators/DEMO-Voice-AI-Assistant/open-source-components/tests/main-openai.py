import openai
from datetime import datetime

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # "http://<Your api-server IP>:port"
    api_key="sk-no-key-required",
)

before = datetime.now()
print(f"before: {before}")
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
        },
        {"role": "user", "content": "Name the planets in the solar system?"},
        {
            "role": "assistant",
            "content": "Of course! There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Would you like me to provide more information on any of these planets?",
        },
        {"role": "user", "content": "Tell me more about earth."},
        {
            "role": "assistant",
            "content": "Sure! Earth is the third planet from the sun in our solar system and is the only known planet to support life. It has a diverse range of ecosystems, including forests, grasslands, deserts, oceans, and polar regions. Earth's atmosphere protects life on its surface, and it has a stable climate that allows for liquid water, which is essential for life as we know it.\nEarth is approximately 93 million miles (150 million kilometers) away from the sun, and it takes 365.24 days to complete one orbit around the sun. Earth's rotation causes day and night cycles, and its gravitational pull holds the planet together and keeps its atmosphere in place.\nEarth is home to a vast array of living organisms, including plants, animals, fungi, and microorganisms. These organisms have evolved over millions of years through natural selection, and they play a crucial role in maintaining the balance of Earth's ecosystems.\nIn addition to supporting life, Earth is also home to many unique features, such as mountains, valleys, canyons, oceans, and rivers. These features are shaped by geological processes, including plate tectonics, weathering, and erosion.\nOverall, Earth is a remarkable planet that supports an incredible diversity of life and ecosystems, making it a special place in the universe. Would you like me to provide more information on any aspect of Earth?",
        },
        {"role": "user", "content": "Tell me more about the ecosystems."},
    ],
)
after = datetime.now()
print(f"after: {after}")

print("\n\n")
difference = after - before
print(f"time: {difference.seconds}")
print("\n\n")

print(completion.choices[0].message)
