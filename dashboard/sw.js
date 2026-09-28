// Service worker del dashboard: lo que permite instalarlo como app en el
// celular. La página sigue siendo la misma que se abre en el navegador.
//
// Regla: un tablero de monitoreo nunca muestra datos viejos como si fueran
// de ahora. Por eso la página y los datos van siempre primero a la red; la
// copia guardada sólo se usa sin conexión, y en ese caso la respuesta lleva
// `x-pulso-guardado` con la hora en que se guardó para que la página lo diga.

const VERSION = "pulso-v1";
const ESTATICOS = [
  "./",
  "manifest.webmanifest",
  "iconos/icono-192.png",
  "iconos/icono-512.png",
  "iconos/maskable-512.png",
  "iconos/apple-touch-icon.png",
  "iconos/favicon-32.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(ESTATICOS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((claves) => Promise.all(claves.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

async function redPrimero(req) {
  const cache = await caches.open(VERSION);
  try {
    const r = await fetch(req);
    if (r.ok) {
      // La hora de guardado viaja con la copia, no en otra tabla aparte.
      const cab = new Headers(r.headers);
      cab.set("x-pulso-guardado", new Date().toISOString());
      // El cuerpo guardado ya va descomprimido: estas cabeceras mentirían.
      cab.delete("content-encoding");
      cab.delete("content-length");
      await cache.put(req, new Response(await r.clone().blob(), { status: r.status, headers: cab }));
    }
    return r;
  } catch (err) {
    const copia = await cache.match(req, { ignoreVary: true });
    if (!copia) throw err;
    return copia;   // ya trae x-pulso-guardado
  }
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  // Datos de Supabase (REST, sólo lectura): red primero, copia sin conexión.
  if (url.hostname.endsWith(".supabase.co") && url.pathname.startsWith("/rest/v1/")) {
    e.respondWith(redPrimero(req));
    return;
  }
  if (url.origin !== self.location.origin) return;

  // La página: red primero, así un deploy nuevo se ve al abrir la app.
  if (req.mode === "navigate") {
    e.respondWith(redPrimero(req).catch(() => caches.match("./")));
    return;
  }
  // Íconos y manifest: no cambian entre deploys de la misma VERSION.
  e.respondWith(caches.match(req).then((r) => r || fetch(req)));
});
