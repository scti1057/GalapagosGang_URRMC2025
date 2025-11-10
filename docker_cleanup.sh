#!/bin/bash

echo "🛑 Stoppe alle laufenden Container..."
docker stop $(docker ps -aq) 2>/dev/null

echo "🧹 Entferne alle Container..."
docker rm $(docker ps -aq) 2>/dev/null

echo "🧼 Entferne alle Images..."
docker rmi $(docker images -q) 2>/dev/null

echo "🪣 Entferne alle Volumes..."
docker volume rm $(docker volume ls -q) 2>/dev/null

echo "🌐 Entferne benutzerdefinierte Netzwerke..."
docker network rm $(docker network ls | grep -v "bridge\|host\|none" | awk 'NR>1 {print $1}') 2>/dev/null

echo "✅ Bereinigung abgeschlossen."
