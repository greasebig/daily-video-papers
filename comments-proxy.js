#!/usr/bin/env node
'use strict';
/*
 * Guestbook comment proxy — listens on port 7897.
 * The browser widget POSTs here; this server adds the GitHub token and
 * appends the comment to comments/data.json in the repo (file-backed store,
 * the same way the paper site commits its content). The token NEVER reaches
 * the browser, so the page never redirects to GitHub.
 *
 * Usage:
 *   GITHUB_TOKEN=ghp_xxx node comments-proxy.js
 *
 * Env:
 *   PORT            (default 7897)
 *   GITHUB_TOKEN    (required) a PAT with repo contents write access
 *   GITHUB_REPO     (default greasebig/greasebig.github.io) owner/repo
 *   GITHUB_DATA     (default comments/data.json) path to the comment store
 *   ALLOW_ORIGIN    (default *) CORS allowed origin for the comment page
 *   HTTPS_PROXY / HTTP_PROXY  (optional) if your network needs an outbound
 *                   proxy to reach api.github.com (e.g. http://127.0.0.1:7890),
 *                   set it here and requests will be tunneled through it.
 */
const http = require('http');
const https = require('https');
const net = require('net');
const tls = require('tls');
const { URL } = require('url');

const PORT = process.env.PORT || 7897;
const TOKEN = process.env.GITHUB_TOKEN || '';
const REPO = process.env.GITHUB_REPO || 'greasebig/greasebig.github.io';
const DATA_PATH = process.env.GITHUB_DATA || 'comments/data.json';
const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || '*';
const API_HOST = 'api.github.com';
// Optional outbound proxy (only used when explicitly set; default = direct).
const OUTBOUND_PROXY = process.env.HTTPS_PROXY || process.env.https_proxy ||
                       process.env.HTTP_PROXY || process.env.http_proxy || '';

function send(res, code, obj, origin) {
  res.writeHead(code, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': origin || ALLOW_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  res.end(JSON.stringify(obj));
}

// One helper for every GitHub REST call. Tunnels through OUTBOUND_PROXY when set.
function githubRequest(method, path, payload, cb) {
  const body = payload ? Buffer.from(JSON.stringify(payload)) : null;
  const headers = {
    'Authorization': 'Bearer ' + TOKEN,
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'gh-comments-proxy'
  };
  if (body) { headers['Content-Type'] = 'application/json'; headers['Content-Length'] = body.length; }

  function onResponse(r) {
    let data = '';
    r.on('data', function (c) { data += c; });
    r.on('end', function () {
      let json; try { json = JSON.parse(data); } catch (e) { json = {}; }
      cb(r.statusCode, json);
    });
  }

  if (OUTBOUND_PROXY) {
    const p = new URL(OUTBOUND_PROXY);
    const tunnel = http.request({
      host: p.hostname,
      port: p.port || 80,
      method: 'CONNECT',
      path: API_HOST + ':443',
      headers: { Host: API_HOST + ':443' }
    });
    tunnel.on('connect', function (res, socket) {
      if (res.statusCode !== 200) { cb(res.statusCode, { error: 'proxy tunnel failed: ' + res.statusCode }); return; }
      const tlsSocket = tls.connect({ socket: socket, servername: API_HOST }, function () {
        const req = https.request({
          host: API_HOST,
          path: path,
          method: method,
          headers: headers,
          createConnection: function () { return tlsSocket; }
        }, onResponse);
        req.on('error', function (e) { cb(0, { error: e.message }); });
        if (body) req.write(body);
        req.end();
      });
      tlsSocket.on('error', function (e) { cb(0, { error: e.message }); });
    });
    tunnel.on('error', function (e) { cb(0, { error: e.message }); });
    tunnel.end();
  } else {
    const req = https.request({ host: API_HOST, path: path, method: method, headers: headers }, onResponse);
    req.on('error', function (e) { cb(0, { error: e.message }); });
    if (body) req.write(body);
    req.end();
  }
}

// GET the current data file (returns {arr, sha} or {arr:[], sha:null} if 404).
function readData(cb) {
  githubRequest('GET', '/repos/' + REPO + '/contents/' + DATA_PATH, null, function (code, json) {
    if (code === 200) {
      let arr = [];
      if (json && json.content) {
        try { arr = JSON.parse(Buffer.from(json.content, 'base64').toString('utf-8')); } catch (e) { arr = []; }
      }
      cb(200, arr, json ? json.sha : null);
    } else if (code === 404) {
      cb(404, [], null);
    } else {
      cb(code, null, null, json);
    }
  });
}

// PUT the updated data file back (sha required unless creating).
function writeData(arr, sha, cb) {
  const content = Buffer.from(JSON.stringify(arr, null, 2), 'utf-8').toString('base64');
  const payload = sha
    ? { message: 'guestbook: add comment', content: content, sha: sha }
    : { message: 'guestbook: add comment', content: content };
  githubRequest('PUT', '/repos/' + REPO + '/contents/' + DATA_PATH, payload, function (code, json) {
    cb(code, json);
  });
}

function postComment(name, body, cb) {
  readData(function (code, arr, sha, errJson) {
    if (code !== 200 && code !== 404) { cb(code, errJson || {}); return; }
    if (!Array.isArray(arr)) arr = [];
    arr.push({
      user: { login: name && name.length ? name : 'guest' },
      created_at: new Date().toISOString(),
      body: body
    });
    writeData(arr, sha, function (wcode, wjson) {
      if (wcode >= 200 && wcode < 300) cb(200, { ok: true });
      else cb(wcode || 502, wjson);
    });
  });
}

const server = http.createServer(function (req, res) {
  const origin = req.headers.origin || '*';
  if (req.method === 'OPTIONS') { send(res, 204, {}, origin); return; }
  if (req.url === '/health') { send(res, 200, { ok: true, repo: REPO, proxy: OUTBOUND_PROXY || 'direct' }); return; }

  if (req.method === 'POST' && req.url === '/api/comment') {
    if (!TOKEN) { send(res, 500, { error: 'GITHUB_TOKEN not set' }, origin); return; }
    let raw = '';
    req.on('data', function (c) { raw += c; if (raw.length > 1e6) req.destroy(); });
    req.on('end', function () {
      let p; try { p = JSON.parse(raw); } catch (e) { send(res, 400, { error: 'bad json' }, origin); return; }
      const name = (p.name || '').toString().trim().slice(0, 40);   // optional; no nickname UI
      const body = (p.body || '').toString().trim().slice(0, 2000);
      if (!body) { send(res, 400, { error: 'body required' }, origin); return; }
      postComment(name, body, function (code, json) {
        if (code >= 200 && code < 300) send(res, 200, { ok: true, comment: json }, origin);
        else send(res, code || 502, { error: (json && json.message) || 'github error', detail: json }, origin);
      });
    });
    return;
  }
  send(res, 404, { error: 'not found' }, origin);
});

server.listen(PORT, function () {
  console.log('Comments proxy listening on http://localhost:' + PORT);
  console.log('Target repo: ' + REPO + (TOKEN ? '  (token: set)' : '  (token: MISSING)'));
  console.log('GitHub egress: ' + (OUTBOUND_PROXY ? 'via ' + OUTBOUND_PROXY : 'direct'));
});
