import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Ce fichier de config vit dans tp1/. On fixe `root` sur son propre dossier
// pour que l'application soit servie depuis tp1/, tout en gardant le
// package.json à la racine du workspace.
const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: here,
  // Chemins relatifs dans la page générée : elle fonctionne quel que soit le
  // dossier où elle est servie (ex. https://<compte>.github.io/<dépôt>/).
  base: "./",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
