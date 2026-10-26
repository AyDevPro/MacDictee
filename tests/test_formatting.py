"""Tests de la mise en forme des dictées : .venv/bin/python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formatting import format_dictation  # noqa: E402


class Listes(unittest.TestCase):
    def test_phrase_reelle(self):
        # transcription Whisper d'une vraie dictée de l'utilisateur
        self.assertEqual(
            format_dictation("Par exemple si je fais point numéro 1 il faut faire ça, point numéro 2 il faut faire "
                             "ça, point numéro 3 il faut faire ça."),
            "Par exemple si je fais :\n1. Il faut faire ça\n2. Il faut faire ça\n3. Il faut faire ça.")

    def test_sans_introduction(self):
        self.assertEqual(format_dictation("Point numéro 1, acheter du pain. Point numéro 2, appeler Paul."),
                         "1. Acheter du pain\n2. Appeler Paul.")

    def test_adverbes(self):
        self.assertEqual(
            format_dictation("Pour demain : premièrement relire le contrat, deuxièmement envoyer le devis et "
                             "troisièmement prévenir Julie."),
            "Pour demain :\n1. Relire le contrat\n2. Envoyer le devis\n3. Prévenir Julie.")

    def test_ordinaux_et_etapes(self):
        self.assertEqual(format_dictation("Premier point le budget. Deuxième point le planning."),
                         "1. Le budget\n2. Le planning.")
        self.assertEqual(format_dictation("Étape 1 ouvrir le capot, étape 2 vérifier l'huile"),
                         "1. Ouvrir le capot\n2. Vérifier l'huile")

    def test_nombres_en_lettres(self):
        self.assertEqual(format_dictation("point numéro un la salle, point numéro deux le traiteur"),
                         "1. La salle\n2. Le traiteur")

    def test_puces(self):
        self.assertEqual(format_dictation("Liste de courses tiret du pain tiret du lait tiret des œufs."),
                         "Liste de courses :\n- Du pain\n- Du lait\n- Des œufs.")

    def test_tirets_ecrits_par_whisper(self):
        # sorties réelles de Whisper pour « tiret » dicté avec des pauses
        self.assertEqual(format_dictation("Liste de course. – Du pain. – Du lait. – Des œufs."),
                         "Liste de course.\n- Du pain\n- Du lait\n- Des œufs.")
        self.assertEqual(format_dictation("Les tâches de la semaine. – Relire le contrat. – Envoyer la facture."),
                         "Les tâches de la semaine.\n- Relire le contrat\n- Envoyer la facture.")

    def test_retours_a_la_ligne(self):
        self.assertEqual(format_dictation("Bonjour Paul, à la ligne. Merci pour ton message. Nouveau paragraphe. "
                                          "À bientôt."),
                         "Bonjour Paul\nMerci pour ton message.\n\nÀ bientôt.")


class RienATouche(unittest.TestCase):
    def test_phrase_normale(self):
        text = "Je t'envoie le compte rendu de la réunion d'hier avant midi."
        self.assertEqual(format_dictation(text), text)

    def test_mention_isolee(self):
        text = "On reviendra sur le point numéro 3 de l'ordre du jour."
        self.assertEqual(format_dictation(text), text)

    def test_numeros_qui_ne_se_suivent_pas(self):
        text = "Le point numéro 2 et le point numéro 5 sont validés."
        self.assertEqual(format_dictation(text), text)

    def test_un_seul_tiret(self):
        text = "C'est un mot avec un tiret au milieu."
        self.assertEqual(format_dictation(text), text)

    def test_tirets_d_incise(self):
        text = "Paul – mon collègue de Lyon – arrive demain, et Julie – sa cheffe – aussi."
        self.assertEqual(format_dictation(text), text)

    def test_mots_proches(self):
        text = "Il est arrivé premier, puis il a fait une deuxième tentative."
        self.assertEqual(format_dictation(text), text)


if __name__ == "__main__":
    unittest.main()
