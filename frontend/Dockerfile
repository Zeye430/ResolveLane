# The web interface: built with Node, served by nginx, which also forwards /api to the API so
# the browser sees one origin, as with the development server.

FROM node:22.23.5-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build


FROM nginx:1.30.6-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/40-demo-accounts.sh /docker-entrypoint.d/40-demo-accounts.sh
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
