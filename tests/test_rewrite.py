"""Déclencheurs et garde-fou de la réécriture IA (sans charger le modèle) :
.venv/bin/python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rewrite import guard, needs_rewrite  # noqa: E402


class Declencheurs(unittest.TestCase):
    def test_utile(self):
        for text in ["alors euh je voulais te dire que le projet avance bien",
                     "on se retrouve à 14h, non pardon à 15h devant la gare",
                     "pour demain d'abord appeler le comptable ensuite envoyer le devis",
                     "il faut que je pense à trois trucs, appeler Paul puis réserver la salle",
                     " ".join(["mot"] * 50)]:
            self.assertTrue(needs_rewrite(text), text)

    def test_inutile(self):
        # phrases propres : collées tout de suite, sans passer par l'IA
        for text in ["Je t'envoie le compte rendu de la réunion d'hier avant midi.",
                     "Est-ce que tu peux me faire un résumé de la réunion de ce matin ?",
                     "Ensuite on verra.",
                     "Point numéro 1 acheter du pain, point numéro 2 appeler Paul."]:  # les règles s'en chargent
            self.assertFalse(needs_rewrite(text), text)


class GardeFou(unittest.TestCase):
    # vraies réponses de modèles relevées au banc d'essai
    def test_bonnes_reecritures(self):
        self.assertTrue(guard(
            "euh donc pour demain faut que je pense à trois trucs, d'abord appeler le comptable, ensuite envoyer le "
            "devis à Julie et enfin réserver le resto pour vendredi",
            "Pour demain, il faut que je pense à trois trucs :\n- Appeler le comptable\n- Envoyer le devis à Julie\n"
            "- Réserver le resto pour vendredi"))
        self.assertTrue(guard("alors euh je voulais te dire que bah en fait le projet avance bien et euh on devrait "
                              "pouvoir livrer vendredi",
                              "Alors je voulais te dire que le projet avance bien et on devrait pouvoir livrer vendredi."))

    def test_correction_volontaire(self):
        self.assertTrue(guard("on se retrouve à 14h, non pardon à 15h devant la gare",
                              "On se retrouve à 15h, devant la gare."))

    def test_phrase_supprimee(self):  # Qwen2.5 3B avait perdu toute la première phrase
        raw = ("Par contre du coup là le main libre, tu vois typiquement là je viens d'appuyer, je reste là, ça marche "
               "nickel. Juste sur Whisperflow ça prenait la place des emojis quand il était lancé, là ça prend pas la "
               "place, ça lance aussi les emojis mais je peux désactiver les emojis, j'en ai pas besoin. Mais ça a "
               "l'air de marcher en tout cas.")
        out = ("Juste sur Whisperflow ça prenait la place des emojis quand il était lancé, là ça prend pas la place, "
               "ça lance aussi les emojis mais je peux désactiver les emojis, j'en ai pas besoin. Mais ça a l'air de "
               "marcher en tout cas.")
        self.assertFalse(guard(raw, out))

    def test_reponse_inventee(self):  # Qwen2.5 1,5B répondait à la question au lieu de la recopier
        self.assertFalse(guard("Est-ce que tu peux me faire un résumé de la réunion de ce matin ?",
                               "Bien sûr ! Voici un résumé : la réunion a porté sur le budget et les priorités du "
                               "trimestre, avec une décision sur le prestataire."))

    def test_sens_change(self):  # « Je t'envoie » devenu « Merci pour »
        self.assertFalse(guard("euh je t'envoie le compte rendu de la réunion d'hier avant midi",
                               "Merci pour le compte rendu de la réunion d'hier."))

    def test_vide(self):
        self.assertFalse(guard("euh bonjour Paul", ""))


if __name__ == "__main__":
    unittest.main()
